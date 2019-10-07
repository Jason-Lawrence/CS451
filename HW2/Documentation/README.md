---
title: "CS451 Gaussian Elimination Parallelization"
output: html_document
---
## By Jason Lawrence
### A20381993
---
## How to run
In the command line navigate to the directory that holds the gauss.c file. Once there type into the command line: gcc -pthreads -o gauss gauss.c
This will create an executable called gauss. To run the program type: ./gauss [N] [seed]. Initially the given serial code is selected. To change
this to one of the parallized functions comment out line 147 in gauss.c and un comment line 148 for the pthread implementation or line 149 for the 
openmp implementation. To run Pthreads in the Command line type: gcc -pthreads -o gauss gauss.c. For OpenMP type: gcc -fopenmp -o gauss gauss.c. 

## Correctness argument
Both the pthreads and openmp algorithms give the same results to the serial version of the algorithm when given the same N value and random seed.
There are no dependencies between row columns since columns are mutated by the row above.

## Versions

### Pthreads
In the Pthreads implementation I initially paralellized the inner most for loop but I was not getting much in performance improvement. I was also dynamically making threads each iteration of the second for loop which also slowed down my algorithm. Eventually, I decided that the simple solution was the most pratical with initializing 4 static threads at the beginning and then creating four threads to split up the work. I played around with the amount of threads created before I settled on four but I found that i wasnt getting any significant performance increases. I think this is due to my CPU having 4 cores and each core only having one thread that it can work on at once with out context switching. Having more threads will cause a lot of overhead and slow down performance. Although, the best result is when I had 16 threads. I think this is due to the OS job scheduler having a higher chance of picking one of my threads to run over one of the hundreds of other jobs I was also running. 

## Performance

### Serial
|N      |Time (ms) |
|:------|:---------|
|500    |98.09     |
|1000   |752.084   |
|1500   |2590.33   |
|2000   |6003.05   |

![Serial](C:\Users\Jason\Desktop\Projects\CS451\HW2\Documentation\Serial.png)

### Pthreads
|N      |Time (ms) |
|:------|:---------|
|500    |129.619   |
|1000   |383.891   |
|1500   |1103.45   |
|2000   |2863.47   |

![Pthreads](C:\Users\Jason\Desktop\Projects\CS451\HW2\Documentation\Pthreads.png)

### OpenMP
|N      |Time (ms)   |
|:------|:-----------|
|500    |66.299      |
|1000   |236.513     |
|1500   |938.538     |
|2000   |1918.22     |

![OpenMP](C:\Users\Jason\Desktop\Projects\CS451\HW2\Documentation\OpenMP.png)

## Comparison
As can be seen from the results serial is the slowest with Pthreads being significantly faster for large N. OpenMP is evan faster even at smaller N values and thus performs the best. this can be easily seen in the following graph.

![Comparison](C:\Users\Jason\Desktop\Projects\CS451\HW2\Documentation\comparison.png)

## Future Work
Things to improve on is in the Pthreads figuring out how to efficiently create threads dynamically that way for large N we could split up the work even more. To do this we would have to also change how what row the thread is suppose to work on as well cause right now it is hardcoded to do evey fourth row. 



