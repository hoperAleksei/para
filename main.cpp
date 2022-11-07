#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>
#include <iostream>
#include <memory.h>
#include <omp.h>
#include <thread>
#include <vector>
#include <mutex>
#include <algorithm>

#define CACHE_LINE 64u
#define N 100000000
#define NN 100

static unsigned g_num_thread = std::thread::hardware_concurrency();

void set_num_threads(unsigned T) {
	g_num_thread = T;
	omp_set_num_threads(T);
}

unsigned get_num_threads() {
	return g_num_thread;
}

struct result_t {
	double value, milliseconds;
};

result_t run_experiment(double (*integrate)(double, double,
                                            double (*f)(double)),
                        double a, double b, double (*f)(double)) {
	result_t res;
	auto tm1 = std::chrono::steady_clock::now();
	
	res.value = integrate(-1, 1, f);
	
	res.milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::steady_clock::now() - tm1)
			.count();
	
	return res;
}

double integrate_seq(double a, double b, double (*f)(double)) {
	double dx = (b - a) / N;
	double res = 0;
	
	for (int i = 0; i < N; i++) {
		res += f(a + i * dx);
	}
	
	return dx * res;
}


double integrate_par(double a, double b, double (*f)(double)) {
	double dx = (b - a) / N;
	double res = 0;
#pragma omp parallel for reduction(+ : res)
	for (int i = 0; i < N; i++) {
		res += f(a + i * dx);
	}
	
	return dx * res;
}

double integrate_rr(double a, double b, double (*f)(double)) {
	double dx = (b - a) / N;
	double res = 0;
	
	unsigned P = omp_get_num_procs();
	double *partical_res;
#pragma omp parallel
	{
//		unsigned T = omp_get_num_threads();
		unsigned T = g_num_thread;
		unsigned t = omp_get_thread_num();

#pragma omp single
		{ partical_res = (double *) calloc(P, sizeof(double)); }
		double sum = 0;
		
		for (unsigned R = 0; t + R * T < N; ++R) {
			sum += f(a + (t + R * T) * dx);
		}
		partical_res[t] += sum;
		
		
	}
	for (unsigned t = 0; t < P; ++t)
		res += partical_res[t];
	
	free(partical_res);
	
	return dx * res;
}

union partial_sum_t {
	double value;
	alignas(double) char pd[64];
};

double average_par_1(const double *V, size_t n) {
	partial_sum_t *sums;
	double r;

#pragma omp parallel
	{
		unsigned T = omp_get_num_threads();
		unsigned t = omp_get_thread_num();
		double local_sum;
#pragma omp single
		{
			sums = (partial_sum_t *) malloc(T * sizeof(partial_sum_t));
			
		};
		for (size_t i = t; i < n; i += T) {
			local_sum += V[i];
			sums[t].value = local_sum;
			
		}
		for (size_t i = 0; i < T; ++i) {
			free(sums);
			r += sums[i].value;
		}
	}
	return r;
}


double average_par_2(const double *V, size_t m) {
	partial_sum_t *sums;
	double r;
	unsigned T;

#pragma omp parallel shared(T, sums)
	{
		unsigned t = omp_get_thread_num();
		double local_sum;

#pragma omp single
		{
			T = omp_get_num_threads();
			sums = (partial_sum_t *) malloc(T * sizeof(partial_sum_t));
		};
		
		unsigned n_t = t < (m % T) ? (int) m / T + 1 : (int) m / T;
		unsigned i_0 = t < (m % T) ? ((int) m / T + 1) * T : (int) m / T * t + (m % T);
		
		for (size_t i = i_0; i < n_t; i++) {
			local_sum += V[i];
		}
		sums[t].value = local_sum;
	};
	
	for (size_t i = 0; i < T; ++i) {
		r += sums[i].value;
	}
	
	free(sums);
	return r;
}

double average_cs_omp(const double *v, size_t n) {
	double total_sum = 0.;

#pragma omp parallel
	{
		unsigned T = omp_get_num_threads();
		unsigned t = omp_get_thread_num();
		size_t nt, i0;
		
		if (t < n % T) {
			nt = n / T + 1;
			i0 = nt * t;
		} else {
			nt = n / T;
			i0 = t * (n / T) + (n % T);
		}
		
		double par_sum;
		
		for (size_t i = i0; i < nt + i0; i++) {
			par_sum += v[i];
		}
#pragma omp critical
		{
			total_sum += par_sum;
		}
	}
	
	return total_sum / n;
}
//double integrate_(double a, double b, double (*f)(double)) {
//
//	double dx = (b - a) / N;
//	unsigned T = std::thread::hardware_concurrency();
//	std::vector<double> partial_res(T);
//	auto thread_proc = [& partial_res, T, a, b, f, dx] (unsigned t)
//	{
//		double &a = (b-a) / N;
//
//	};
//}

#define load(N) for(auto i=N; i>0; --i) std::cout << "load\n";

double average_cs_cpp(const double *V, size_t m) {
	std::mutex mtx1, mtx2;
	int x1 = 0, x2 = 0;
	auto tp1 = [&mtx1, &x1, &mtx2, &x2]() {
		mtx1.lock();
		x1++;
		
		load(100);
		mtx2.lock();
		x2++;
		
		mtx2.unlock();
		mtx1.unlock();
	};
	
	auto tp2 = [&mtx2, &x2, &mtx1, &x1]() {
		mtx2.lock();
		x2--;
		mtx1.lock();
		x1--;
		
		mtx1.unlock();
		mtx2.unlock();
	};
	
	auto th1 = std::thread(tp1), th2 = std::thread(tp2);
	th1.join();
	th2.join();
	std::cout << "x1 = " << x1 << ", x2 = " << x2 << std::endl;
}

double integrate_cpp_partial_sums(double a, double b, double (*f)(double)) {
//	std::size_t T = std::thread::hardware_concurrency();
	std::size_t T = g_num_thread;
	
	auto partial_sums = std::make_unique<double[]>(T);
	
	auto thread_proc = [T, &partial_sums, a, b, f](std::size_t t) {
		double sum = 0;
		partial_sums[t] = 0;
		auto dx = (b - a) / N;
		for (auto i = t; i < N; i += T)
			sum += f(i * dx + a);
		
		partial_sums[t] += sum;
		partial_sums[t] *= dx;
	};
	std::vector<std::thread> workers;
	for (std::size_t t = 0; t < T; ++t)
		workers.emplace_back(thread_proc, t);
	for (auto &worker: workers)
		worker.join();
	for (std::size_t t = 1; t < T; ++t)
		partial_sums[0] += partial_sums[t];
	
	return partial_sums[0];
}

// беззнаковое целое число
#define CACHE_LINE 64u

// Выделяйте память под структуры по выровненному адресу:
struct partialSumT {
	double value[CACHE_LINE / sizeof(double)];
};

double integrateArrAlign(double a, double b, double (*f)(double)) {
	unsigned T;
	double result = 0, dx = (b - a) / N;
	partialSumT *accum = nullptr;

#pragma omp parallel shared(accum, T)
	{
		auto t = omp_get_thread_num();
#pragma omp single
		{
			T = (unsigned) omp_get_num_threads();
			accum = new partialSumT[T];
		}
		
		for (unsigned i = t; i < N; i += T) {
			accum[t].value[0] += f(dx * i + a);
		}
	}
	
	for (unsigned i = 0; i < T; ++i) {
		result += accum[i].value[0];
	}
	
	delete[] accum;
	
	return N;
}


//void measure_scalability(auto integrate_fn) {
//	auto P = omp_get_num_procs();
//	auto partial_res = std::make_unique<result_t[]>(P);
//	auto f = [](double x) { return x * x; };
//
//	for (auto T = 1; T <= P; ++T) {
//		set_num_threads(T);
//		partial_res[T - 1] = run_experiment(integrate_fn, -1, 1, f);
//		auto speedup = partial_res[0].milliseconds / partial_res[T - 1].milliseconds;
//		std::cout << T << ',' << partial_res[T - 1].milliseconds << ',' << speedup << ',' << partial_res[T - 1].value
//		          << '\n';
//
//	}
//
//}

void fillVector(double *v, size_t n) {
	for (size_t i = 0; i < n; ++i) {
		v[i] = 1.0;
	}
}

result_t run_experiment(double (*average)(const double *, size_t), const double *v, size_t n) {
	auto tm1 = std::chrono::steady_clock::now();
	double value = average(v, n);
	auto time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tm1).count();
	result_t res{value, (double) time};
	return res;
}

void measure_scalability(auto averageFunction) {
	auto P = omp_get_num_procs();
	auto partial_res = std::make_unique<result_t[]>(P);
	auto v = std::make_unique<double[]>(N);
	fillVector(v.get(), N);
	for (auto T = 1; T <= P; ++T) {
		set_num_threads(T);
		partial_res[T - 1] = run_experiment(averageFunction, v.get(), N);
		auto speedup = partial_res[0].milliseconds / partial_res[T - 1].milliseconds;
		std::cout << "threads: " << T << std::endl;
		std::cout << "time: " << partial_res[T - 1].milliseconds << std::endl;
		std::cout << "value: " << partial_res[T - 1].value << std::endl;
		std::cout << "speedup: " << speedup << std::endl << std::endl;
	}
}

double average (const double *V, size_t m) {
	double result_t = 0;

#pragma omp parallel shared(result_t)
	{
		unsigned T =omp_get_num_threads();
		unsigned t = omp_get_thread_num();
		size_t nt = m/T, i0 = m % T;
		
		if (t<i0)
			i0 = ++nt*t;
		else
			i0 = t*nt;
		
		double par_sum = 0;
		
		for (size_t i = i0; i < nt; i++) {
			par_sum += V[i];
		}

#pragma omp critical
		{
			result_t += par_sum;
		}
	}
	
	return result_t / m;
}

extern int global;
std::mutex mtx;

void thread_proc(int x)
{
	int x_sg = x * x;
	x_sg = x_sg + x_sg + x * x;
	{
		std::scoped_lock lock{mtx};
		++global;
	}
	
//	return x_sg;
	return;
}

//queue<int> q;
//bool q_empty;
//
//auto reader()
//{
//	while (q_empty)
//		continue;
//	int val = q.pop();
//
//	q_empty = true;
//
//	return val;
//}
//
//auto writer(int x)
//{
//	q.push(x);
//
//	q_empty = false;
//}



int main() {
//	auto f = [](double x) { return x * x; };
//	auto r_seq = run_experiment(integrate_seq, -1, 1, f);
//	auto r_par = run_experiment(integrate_par, -1, 1, f);
//	auto r_rr = run_experiment(integrate_rr, -1, 1, f);
//	auto r_cps = run_experiment(integrate_cpp_partial_sums, -1, 1, f);
//	auto r_arr = run_experiment(integrateArrAlign, -1, 1, f);
//
//	std::cout << "Res seq: t=" << r_seq.milliseconds << ", r=" << r_seq.value
//	          << '\n';
//	std::cout << "Res par: t=" << r_par.milliseconds << ", r=" << r_par.value
//	          << '\n';
//	std::cout << "Res rr: t=" << r_rr.milliseconds << ", r=" << r_rr.value
//	          << '\n';
////	std::cout << "Res lol: t=" << r_lol.milliseconds << ", r=" << r_lol.value
////	          << '\n';
//	std::cout << "Res cps: t=" << r_cps.milliseconds << ", r=" << r_cps.value
//	          << '\n';
//	std::cout << "Res arr: t=" << r_arr.milliseconds << ", r=" << r_arr.value
//	          << '\n';
//	measure_scalability(integrate_cpp_partial_sums);
//	measure_scalability(integrate_rr);
//	measure_scalability(integrateArrAlign);


//	std::cout << average_par_1(V, 9) << std::endl;
//	std::cout << average_par_2(V, 9) << std::endl;
//	std::cout << average_cs_omp(V, 9) << std::endl;

//    std::cout << "AveragePar1:" << std::endl;
//    measure_scalability(average_par_1);
//	measure_scalability(average_cs_cpp);
//	std::cout << "AveragePar2:" << std::endl;
//	measure_scalability(average_par_2);
//	std::cout << "CriticalSection:" << std::endl;
//	measure_scalability(average_cs_omp);
	
	std::size_t n = 100000;
	auto m = std::make_unique<double[]>(n);
	std::generate_n(m.get(), n, []() {
		static int i;
		return i++;
	});
	
	std::cout << "Average value: " << average(m.get(), n) << std::endl;
	
	return 0;
	
	
	
}