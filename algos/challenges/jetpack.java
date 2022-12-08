/**
Difficulty: Hard

Time Complexity:
Q * log(N) * N

Required Knowledge: Binary Search, Greedy

At the core of this solution is the subproblem of, given a number of canisters and a maximum hop distance, can you reach the mainland from the starting island.

It can be shown that you can solve this subproblem greedily, meaning that if you are on an island at position 10 and you can reach islands at positions 12 and 14 in one jump, it is always better to jump straight to the island at position 14 than the one at position 12. More formally, you want each hop to move you to the furthest island whose distance from the current island does not exceed the max hop distance. This means that you can implement answer this question by one linear scan of the islands, updating the number of hops you've taken as you go.

Now that you have a function(numCanisters, maxHopDistance)->boolean, you can answer each query. Each query gives you the number of canisters and wants you to find the smallest hop distance for which this function returns true.

The simple solution is to just try all possible hop distances, starting from 1, until you find one that returns true. There are a couple things you can do to improve this.

The first is to change your bounds on your search. If you have C canisters, and the total distance from the start island to the mainland is D, you know that any hop distance that is lower than D/C could not possibly make it to the mainland, so you can start your search at D/C instead of 1.

You can also say that the maximum hop distance answer is just D, since you can trivially make it with D hop distance with 1 canister.

The trick to passing all of the test cases is to realize you can do a binary search on the hop distance instead of a linear search. Your starting range is (D/C, D), meaning that this binary search will run at most log(D) iterations.

This means that each query runs in I * log(D) time, where I is the number of islands, so the total runtime is Q * I * log(D)
 */

import java.util.*;

public class Jetpack {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int N = in.nextInt();
        int[] arr = new int[N];
        for (int i = 0; i < N; i++) {
            arr[i] = in.nextInt();
        }

        int Q = in.nextInt();
        while(Q-- > 0) {
            int jumps = in.nextInt();
            int minDistPossible = (arr[N-1] - arr[0]) / jumps;
            int maxDistPossible = arr[N-1] - arr[0];
            while (minDistPossible < maxDistPossible) {
                int midDist = (minDistPossible + maxDistPossible) / 2;
                if (simulate(arr, midDist) > jumps) {
                    minDistPossible = midDist + 1;
                } else {
                    maxDistPossible = midDist;
                }
            }
            // do a little linear search at the end to make sure we got the right endpoint
            int dist = Math.min(maxDistPossible, minDistPossible);
            while (simulate(arr, dist) > jumps) {
                dist++;
            }
            System.out.println(dist);
        }

    }
    
    public static int simulate(int[] arr, int dist) {
        int cur = 0;
        int hops = 0;
        while (true) {
            hops++;
            int next = nextIsland(arr, cur, dist);
            if (next == arr.length - 1) {
                // we made it across
                break;
            }
            if (cur == next) {
                // if we couldn't make any progress, we cannot make it across
                return Integer.MAX_VALUE;
            }
            cur = next;
        }
        return hops;
    }
    
    
    public static int nextIsland(int[] arr, int island, int jumpDistance) {
        if (island >= arr.length) return arr.length - 1;
        int startPoint = arr[island];
        while (island < arr.length && arr[island] <= startPoint + jumpDistance) {
            island++;
        }
        return island - 1;
    }
}