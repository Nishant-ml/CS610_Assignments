#include<iostream>
#include<mutex>
#include<queue>
#include<vector>
#include<string>
#include<random>
#include<condition_variable>
#include<atomic>
#include<fstream>
#include<thread>
#include<unistd.h>
#include<string>
using namespace std;

mutex mtxInput,mtxBuffer,mtxProducer,mtxOutput;               
queue<vector<string>> queueBuffer;

condition_variable condVar;
atomic<bool> allFinished(false);
ifstream fin;
ofstream fout;
int bufCapacity;             
size_t linesUsed = 0;


// Producer
void producerTask(int tid,int lmax,int lmin){
    vector<string> tempLines;

    while (true) {
        {
            lock_guard<mutex> lock(mtxInput);
            int l=lmin+(rand())%(lmax-lmin+1);
            tempLines.clear();
            string line;
            for (int i = 0; i < l && getline(fin, line); i++) {
                // line=line+to_string(i);//checking
                tempLines.push_back(move(line));
            }
        }

        if (tempLines.empty()) { 
            break;
        }
        unique_lock<mutex> prod_lock(mtxProducer);
        size_t pos = 0;
        while (pos < tempLines.size()) {
            const size_t remain = tempLines.size() - pos;
            const size_t grab = min(static_cast<size_t>(bufCapacity), remain);

            unique_lock<mutex> buf_lock(mtxBuffer);
            condVar.wait(buf_lock, [&]{
                return linesUsed + grab <= static_cast<size_t>(bufCapacity);
            });

            // Enqueue subchunk atomically
            vector<string> chunk(tempLines.begin() + pos,tempLines.begin() + pos + grab);
            queueBuffer.push(move(chunk));
            linesUsed += grab;
            buf_lock.unlock();
            condVar.notify_all(); 
            pos += grab;
        }
    }
}

bool isFirst=true;

// Consumer
void consumerTask()
{
    while (!allFinished.load() || !queueBuffer.empty())
    {
        vector<string> chunk;
        {
            unique_lock<mutex> lock(mtxBuffer);
            condVar.wait(lock, [&]{
                return !queueBuffer.empty() || allFinished.load();
            });

            if (!queueBuffer.empty()) {
                chunk = move(queueBuffer.front());
                queueBuffer.pop();
                linesUsed -= chunk.size();  // free txt-capacity
                lock.unlock();
                condVar.notify_all(); // wake producers waiting for space
            }
        }

        // Write this chunk atomically to output
        if (!chunk.empty()) {
            lock_guard<mutex> out_lock(mtxOutput);
            for (const auto& s : chunk) {
                if(!isFirst)
                fout<<'\n';
                fout << s;
                isFirst=false;
            }
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc != 7) {
        cerr << "Usage: " << argv[0]
                  << " <input_file> <producer_threads> <lmin> <lmax> <bufferSize_in_lines> <output_file>\n";
        return 1;
    }

    string input_file = argv[1];
    int producers = stoi(argv[2]);
    int lmin = stoi(argv[3]),lmax =stoi(argv[4]);
    bufCapacity = stoi(argv[5]);
    string output_file  = argv[6];

    fin.open(input_file);
    if (!fin) {
        cerr << "Unable to open input file\n";
        return 1;
    }
    fout.open(output_file);
    if (!fout) {
        cerr << "Unable to open output file\n";
        return 1;
    }

    vector<thread> pthreads;
    pthreads.reserve(producers);
    for (int tid = 1; tid <= producers; ++tid) {
        pthreads.emplace_back(producerTask, tid,lmax,lmin);
    }
    vector<thread> cthreads;
    int consumers = max(1, producers / 2); 
    for (int i = 0; i < consumers; ++i) {
        cthreads.emplace_back(consumerTask);
    }
    for (auto &th : pthreads) th.join();//join producers

    allFinished.store(true);
    condVar.notify_all();

    // join consumers
    for (auto &th : cthreads) th.join();

    fin.close();
    fout.close();
    return 0;
}