/**
Difficulty: Medium

Time Complexity:
O(N * log(N))

Required Knowledge: Segment Trees

This simple solution runs in N^2 worst case time and will get 25% credit.

Getting 100% credit requires a solution whose total running time is asymptotically lower than N^2.

Both of the solutions we had generalized the above ideas to group buildings into ranges, and do operations on those ranges of buildings. To do this, each range should keep track of the 3 values (width, height, area covered) in a range. Then, to get the value across two ranges, you can merge the ranges in constant time:

The simplest approach using these is to choose a range size B, and then group each consecutive set of B buildings into ranges. So if B was 100, you'd have the ranges 0-99,100-199, etc... You can also trivially convert each building into a range of 1. So, to perform a query from L to R, we need to merge all of the ranges we have between L and R. So for example if our query is from 70 to 1130, we'd merge all of the 1 building ranges from 70 to 99, merge all of the 100 building ranges from 100-199 to 1000-1099, and merge all of the 1 building ranges from 1100 to 1130. Since at each endpoint we have to merge at most B 1 building ranges, and there are at most N/B combined ranges, the runtime of a query is O(B + N/B).
This is optimal when B = sqrt(N), since then you have sqrt(N) + N/sqrt(N) = sqrt(N) + sqrt(N). If B is any lower than sqrt(N), the N/B term outweights the B term, any larger and the B term outweights the N/B term.
Since each query runs in at most O(sqrt(N)) time, the total time complexity is . This will pass all of the test cases.

However, you can actually do better, using a [Segment Tree Data Structure](http://www.geeksforgeeks.org/segment-tree-set-1-sum-of-given-range/). Segment trees are a data structure designed around querying ranges of data, and will give you logN runtime for each query, making the overall time complexity O(NlogN). You could actually consider the bucketing solution as a two-level segment tree with a branching of sqrtN instead of 2.
*/

public class Skyline {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int N = in.nextInt();
        int[] w = new int[N];
        int[] h = new int[N];
        for (int i = 0; i < N; i++) {
            w[i] = in.nextInt();
            h[i] = in.nextInt();
        }
        Node root = new Node(w, h, 0, N-1);
        int M = in.nextInt();
        while (M-- > 0) {
            int L = in.nextInt();
            int R = in.nextInt();
            Values v = root.query(L, R);
            if (v.areaCovered == v.width * v.height) {
                System.out.println("1.000");
            } else {
                    double result = (1.0 * v.areaCovered) / (v.width * v.height);
                System.out.printf("%.3f\n", result);
            }
        }
    }

    private static class Values {
        int width;
        int height;
        int areaCovered;
        Values() {
            width = 0;
            height = 0;
            areaCovered = 0;
        }

        Values(int w, int h, int a) {
            this.width = w;
            this.height =  h;
            this.areaCovered = a;
        }

        static Values merge(Values a, Values b) {
            return new Values(a.width + b.width, Math.max(a.height, b.height), a.areaCovered + b.areaCovered);
        }
    }

    private static class Node {
        Values v;
        int l;
        Node lNode;
        int r;
        Node rNode;
        
        Node(int[] w, int[] h, int l, int r) {
            this.l = l;
            this.r = r;
            if (l == r) {
                lNode = null;
                rNode = null;
                v = new Values(w[l], h[l], w[l] * h[l]);
            } else {
                int m = (l + r) / 2;
                lNode = new Node(w, h, l, m);
                rNode = new Node(w, h, m+1, r);
                v = Values.merge(lNode.v, rNode.v);
            }

        }

        Values query(int ql, int qr) {
            if (ql > r || qr < l) return new Values();
            if (ql == l && qr == r) {
                return v;
            }
            int m = (l + r) / 2;
            Values lv = lNode.query(ql, Math.min(m, qr));
            Values rv = rNode.query(Math.max(m+1, ql), qr);
            return Values.merge(lv, rv);
        }
    }
}