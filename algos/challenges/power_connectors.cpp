/**
This problem has a straightforward naive solution: try all N possible offsets, and count the number of matching pins for each. The only catch is that when the connectors are plugged in, one of them has to be rotated 180 degrees, so the pin numbers go in opposite directions. This solution runs in O(n^2) time, which is enough to get partial credit.

In order to solve the problem more efficiently, we need some mathematics. Let's represent each connector as a vector of 0 and 1 values. The amount of current that flows is just the dot product of the two vectors, after suitably flipping and shifting one of them.

The operation of computing this product for all possible offsets has a name -- convolution. This a very common operation in digital signal processing; for example, blurring an image and adding reverberation to a an audio signal are both examples of convolution. In particular, we want to perform circular convolution, which is when we apply this operation to a periodic signal, such that when we shift the arguments relative to each other, the values wrap around.

It turns out that there's a close mathematical connection between convolution and the Fourier transform. For continuous signals, if you take the product of the Fourier transforms of two signals, you get the Fourier transform of their convolution; this is known as the convolution theorem. The same property holds for the Discrete Fourier transform, which applies to vectors of discrete sample points. So we can use the Cooley-Tukey fast Fourier transform, and its inverse, to compute the convolution in O(n log n) using a divide-and-conquer algorithm.

Deriving and fully understanding the FFT algorithm is fairly subtle, but the implementation is fairly short, and you can find plenty of references for how to write the code. However, there are a couple of implementation details that come up when applying it to this problem. First of all, the FFT is defined in terms of complex vectors, so you need to be able to perform addition, multiplication and exponentiation on complex numbers. The output will be a vector of real numbers that can be rounded to the nearest integer. (In principle, doing the FFT with floating-point types can lead to imprecise results, but double-precision gives us roughly 15 significant digits of accuracy, which is plenty.)

Theoretically, the discrete Fourier transform can be applied to periodic signals of any integer length N. However, the Cooley-Tukey FFT algorithm only works when N is a power of 2. (There are other FFT algorithms that are more general, but they're significantly more complicated.) So the easiest solution is simply to pad the inputs to the next greater power of two. This means that the transform won't automatically wrap around correctly; the solution is to concatenate one of the vectors with another copy of itself. Then we just have to take care to only look at the region of the output that corresponds to a valid shift, such that the shorter vector overlaps entirely with the longer one and not the padding.
 */

// Here's a "naive" O(n^2) solution:

#include <iostream>
#include <vector>
#include <sstream>

using namespace std;

void read_case(string& line, vector<bool>& x) {
    istringstream iss(line);
    int pos;
    while (iss >> pos) {
        x[pos-1] = true;
    }
}
    
int main() {
    int n;
    cin >> n;

    vector<bool> a(n), b(n);
    string line;

    getline(cin, line);
    getline(cin, line);
    read_case(line, a);
    getline(cin, line);
    read_case(line, b);

    int best = 0;
    for (int offset = 0; offset < n; offset++) {
        int match = 0;
        for (int i = 0; i < n; i++) {
            int j = (n-i+offset) % n;
            if (a[i] && b[j]) match++;
        }
        best = max(best, match);
    }
    cout << best << endl;
}
And an O(n log n) solution using FFT:

#include <iostream>
#include <cmath>
#include <vector>
#include <complex>

using namespace std;

typedef complex<double> cplx;

const double pi = acos(-1);

void fft(const vector<cplx>& input, int N, int inStart, int s, vector<cplx>& output, int outStart, int level) {
    if (N == 1) {
        output[outStart] = input[inStart];
    } else {
        fft(input, N/2, inStart, s*2, output, outStart, level+1);
        fft(input, N/2, inStart+s, s*2, output, outStart+N/2, level+1);

        for (int k = 0; k < N/2; k++) {
            cplx shift = exp(-2*pi*cplx(0,1)*(double)k/(double)N);
            cplx t = output[outStart+k];
            output[outStart+k] = t + shift*output[outStart+k+N/2];
            output[outStart+k+N/2] = t - shift*output[outStart+k+N/2];
        }
    }
}

int nextPowerOfTwo(int x) {
    int power = 1;
    while (power < x) power *= 2;
    return power;
}

void read_case(string& line, vector<bool>& x) {
    istringstream iss(line);
    int pos;
    while (iss >> pos) {
        x[pos-1] = true;
    }
}

int main() {
    int n;
    cin >> n;

    vector<bool> a(n), b(n);
    string line;

    getline(cin, line);
    getline(cin, line);
    read_case(line, a);
    getline(cin, line);
    read_case(line, b);

    int N = nextPowerOfTwo(n*2);
    vector<cplx> a_(N), b_(N), fftA(N), fftB(N), fftResult(N), result(N);
    for (int i = 0; i < a.size(); i++) {
        if (a[i]) a_[i] = a_[i+n] = 1;
        if (b[i]) b_[i] = 1;
    }

    fft(a_, N, 0, 1, fftA, 0, 0);
    fft(b_, N, 0, 1, fftB, 0, 0);

    // the inverse DFT can be computed using the same algorithm,
    // with slightly modified inputs and outputs:
    //
    //     DFT^-1(x) = conj(DFT(conj(x))) / N
    //
    // where conj is complex conjugation, i.e. taking the negative of the
    // imaginary part.

    for (int i = 0; i < N; i++) {
        fftResult[i] = conj(fftA[i]*fftB[i]);
    }
    fft(fftResult, N, 0, 1, result, 0, 0);
    for (int i = 0; i < N; i++) {
        result[i] = conj(result[i]) / (double) N;
    }

    double best = 0;
    for (int i = n; i < 2*n; i++) {
        best = max(best, real(result[i]));
    }

    cout << (int) (best+0.5) << endl;
}

// And an O(n log n) solution using FFT:

#include <iostream>
#include <cmath>
#include <vector>
#include <complex>

using namespace std;

typedef complex<double> cplx;

const double pi = acos(-1);

void fft(const vector<cplx>& input, int N, int inStart, int s, vector<cplx>& output, int outStart, int level) {
    if (N == 1) {
        output[outStart] = input[inStart];
    } else {
        fft(input, N/2, inStart, s*2, output, outStart, level+1);
        fft(input, N/2, inStart+s, s*2, output, outStart+N/2, level+1);

        for (int k = 0; k < N/2; k++) {
            cplx shift = exp(-2*pi*cplx(0,1)*(double)k/(double)N);
            cplx t = output[outStart+k];
            output[outStart+k] = t + shift*output[outStart+k+N/2];
            output[outStart+k+N/2] = t - shift*output[outStart+k+N/2];
        }
    }
}

int nextPowerOfTwo(int x) {
    int power = 1;
    while (power < x) power *= 2;
    return power;
}

void read_case(string& line, vector<bool>& x) {
    istringstream iss(line);
    int pos;
    while (iss >> pos) {
        x[pos-1] = true;
    }
}

int main() {
    int n;
    cin >> n;

    vector<bool> a(n), b(n);
    string line;

    getline(cin, line);
    getline(cin, line);
    read_case(line, a);
    getline(cin, line);
    read_case(line, b);

    int N = nextPowerOfTwo(n*2);
    vector<cplx> a_(N), b_(N), fftA(N), fftB(N), fftResult(N), result(N);
    for (int i = 0; i < a.size(); i++) {
        if (a[i]) a_[i] = a_[i+n] = 1;
        if (b[i]) b_[i] = 1;
    }

    fft(a_, N, 0, 1, fftA, 0, 0);
    fft(b_, N, 0, 1, fftB, 0, 0);

    // the inverse DFT can be computed using the same algorithm,
    // with slightly modified inputs and outputs:
    //
    //     DFT^-1(x) = conj(DFT(conj(x))) / N
    //
    // where conj is complex conjugation, i.e. taking the negative of the
    // imaginary part.

    for (int i = 0; i < N; i++) {
        fftResult[i] = conj(fftA[i]*fftB[i]);
    }
    fft(fftResult, N, 0, 1, result, 0, 0);
    for (int i = 0; i < N; i++) {
        result[i] = conj(result[i]) / (double) N;
    }

    double best = 0;
    for (int i = n; i < 2*n; i++) {
        best = max(best, real(result[i]));
    }

    cout << (int) (best+0.5) << endl;
}