#include <iostream>
#include <cmath>
#include <complex>
#include <fftw3.h>
#include <vector>
#include <easyx.h>
#include <format>
#include <omp.h>
#include <ctime>

using namespace std;
constexpr int nx = 256;
constexpr int ny = 256;

constexpr double re = 50;
constexpr double PI = 3.1415926;

using v1d = vector<double>;
using v2d = vector<vector<double>>;
using v1c = vector<complex<double>>;
using v2c = vector<vector<complex<double>>>;

void compute_residual(int nx, int ny, double dx, double dy, const v2d& f, v2d& un, v2d& r)
{
	for (int j = 1; j < ny; ++j) {
		for (int i = 1; i < nx; ++i) {
			double d2udx2 = (un[i + 1][j] - 2 * un[i][j] + un[i - 1][j]) / (dx * dx);
			double d2udy2 = (un[i][j + 1] - 2 * un[i][j] + un[i][j - 1]) / (dy * dy);
			r[i][j] = f[i][j] - d2udx2 - d2udy2;
		}
	}
}

void gauss_seidel_mg(int nx, int ny, double dx, double dy, const v2d& f, v2d& un, int V = 4) 
{
	static const double den = 1 / ((-2.0 / (dx * dx) - 2.0 / (dy * dy)) * dx * dx);

	for (int it = 0; it < V; ++it) 
	{
#pragma omp parallel for schedule(static)
		for (int j = 1; j < ny; ++j)
		{
#pragma omp simd
			for (int i = 1 + (j % 2); i < nx; i += 2)
			{
				const double residual = f[j][i] * dx * dx
					- (un[j + 1][i] - 2 * un[j][i] + un[j - 1][i])
					- (un[j][i + 1] - 2 * un[j][i] + un[j][i - 1]);
				un[j][i] += residual * den;
			}

#pragma omp simd
			for (int i = 1 + ((j + 1) % 2); i < ny; i += 2)
			{
				const double residual = f[j][i] * dy * dy
					- (un[j + 1][i] - 2 * un[j][i] + un[j - 1][i])
					- (un[j][i + 1] - 2 * un[j][i] + un[j][i - 1]);
				un[j][i] += residual * den;
			}
		}
	}
}

void restriction(int nxf, int nyf, int nxc, int nyc, const v2d& r, v2d& ec)
{
	for (int i = 1; i < nxc; i++)
	{
		for (int j = 1; j < nyc; j++)
		{
			double x1 = 4 * r[2 * i - 1][2 * j - 1];
			double x2 = 2 * (r[2 * i - 1][2 * j] + r[2 * i - 1][2 * j - 2] + r[2 * i][2 * j - 1] + r[2 * i - 2][2 * j - 1]);
			double x3 = r[2 * i][2 * j] + r[2 * i][2 * j - 2] + r[2 * i - 2][2 * j] + r[2 * i - 2][2 * j - 2];
			ec[i][j] = (x1 + x2 + x3) / 16;
		}
	}

	for (int i = 0; i <= nxc; i++)
	{
		ec[0][i] = r[0][2 * i];
		ec[nyc][i] = r[nyf][2 * i];
	}
	for (int j = 0; j <= nyc; j++)
	{
		ec[j][0] = r[2 * j][0];
		ec[j][nxc] = r[2 * j][nyf];
	}
}

void prolongation(int nxc, int nyc, int nxf, int nyf, const v2d& unc, v2d& ef)
{
	for (int j = 0; j < nyc; ++j)
	{
		for (int i = 0; i < nxc; ++i)
		{
			int fi = 2 * i + 1;
			int fj = 2 * j + 1;
			ef[fi][fj] = unc[i][j];
			ef[fi][fj + 1] = 0.5 * (unc[i][j] + unc[i][j + 1]);
			ef[fi + 1][fj] = 0.5 * (unc[i][j] + unc[i + 1][j]);
			ef[fi + 1][fj + 1] = 0.25 * (unc[i][j] + unc[i][j + 1] + unc[i + 1][j] + unc[i + 1][j + 1]);
		}
	}

	for (int j = 0; j <= nyc; ++j) {
		int fj = 2 * j;
		ef[0][fj] = unc[0][j];       // Bottom boundary (i=0)
		ef[nxf][fj] = unc[nxc][j];   // Top boundary (i=nxf)
	}

	for (int i = 0; i <= nxc; ++i) {
		int fi = 2 * i;
		ef[fi][0] = unc[i][0];       // Left boundary (j=0)
		ef[fi][nyf] = unc[i][nyc];   // Right boundary (j=nyf)
	}
}

void mgN(double dx, double dy, int nlevel, const v2d& f, v2d& un) {

	vector<int> lnx(nlevel, nx), lny(nlevel, ny);
	vector<double> ldx(nlevel, dx), ldy(nlevel, dy);
	for (int i = 1; i < nlevel; i++)
	{
		lnx[i] = lnx[i - 1] / 2;
		lny[i] = lny[i - 1] / 2;
		ldx[i] = ldx[i - 1] * 2;
		ldy[i] = ldy[i - 1] * 2;
	}

	vector<v2d> nu(nlevel), nf(nlevel), nr(nlevel);
	for (int i = 0; i < nlevel; i++)
	{
		nu[i] = v2d(lnx[i] + 1, vector<double>(lny[i] + 1, 0.0));
		nf[i] = v2d(lnx[i] + 1, vector<double>(lny[i] + 1, 0.0));
		nr[i] = v2d(lnx[i] + 1, vector<double>(lny[i] + 1, 0.0));
	}

	nf[0] = f; nu[0] = un;
	for (auto& x : nf[0])
		for (auto& i : x)i = -i;

	for (int i = 0; i < nlevel - 1; i++)//向下
	{
		gauss_seidel_mg(lnx[i], lny[i], ldx[i], ldy[i], nf[i], nu[i]);
		compute_residual(lnx[i], lny[i], ldx[i], ldy[i], nf[i], nu[i], nr[i]);
		restriction(lnx[i], lny[i], lnx[i + 1], lny[i + 1], nr[i], nf[i + 1]);
	}
	gauss_seidel_mg(
		lnx[nlevel - 1], lny[nlevel - 1], ldx[nlevel - 1], ldy[nlevel - 1],
		nf[nlevel - 1], nu[nlevel - 1]);

	for (int i = nlevel - 1; i > 0; i--)//向上
	{
		prolongation(lnx[i], lny[i], lnx[i - 1], lny[i - 1], nu[i], nr[i - 1]);
		for (int j = 1; j < lny[i - 1]; j++)
			for (int k = 1; k < lnx[i - 1]; k++)
				nu[i - 1][j][k] += nr[i - 1][j][k];
		gauss_seidel_mg(lnx[i - 1], lny[i - 1], ldx[i - 1], ldy[i - 1], nf[i - 1], nu[i - 1]);
	}

	for (int i = 0; i <= nx; i++)
		for (int j = 0; j <= ny; j++)un[i][j] = nu[0][i][j];
}

void boundCond(v2d& w)
{
	for (int j = 0; j < ny + 2; ++j) w[0][j] = w[nx][j];
	for (int i = 0; i < nx + 2; ++i) w[i][0] = w[i][ny];
	for (int j = 0; j < ny + 2; ++j) w[nx + 1][j] = w[1][j];
	for (int i = 0; i < nx + 2; ++i) w[i][ny + 1] = w[i][1];
}

void rhs(double dx, double dy, const v2d& w, const v2d& s, v2d& r)
{
	for (int j = 1; j < ny; j++)
	{
		for (int i = 1; i < nx; i++)
		{
			double j1 =
				(w[j][i + 1] - w[j][i - 1]) * (s[j + 1][i] - s[j - 1][i]) -
				(w[j + 1][i] - w[j - 1][i]) * (s[j][i + 1] - s[j][i - 1]);
			double j2 =
				(w[j][i + 1]) * (s[j + 1][i + 1] - s[j - 1][i + 1]) -
				(w[j][i - 1]) * (s[j + 1][i - 1] - s[j - 1][i - 1]) -
				(w[j + 1][i]) * (s[j + 1][i + 1] - s[j + 1][i - 1]) +
				(w[j - 1][i]) * (s[j - 1][i + 1] - s[j - 1][i - 1]);
			double j3 =
				w[j + 1][i + 1] * (s[j + 1][i] - s[j][i + 1]) -
				w[j - 1][i - 1] * (s[j][i - 1] - s[j - 1][i]) -
				w[j + 1][i - 1] * (s[j + 1][i] - s[j][i - 1]) +
				w[j - 1][i + 1] * (s[j][i + 1] - s[j - 1][i]);

			double jac = (j1 + j2 + j3) / (12 * dx * dy);
			r[j][i] =
				(w[j][i + 1] - 2 * w[j][i] + w[j][i - 1]) / (dx * dx * re) +
				(w[j + 1][i] - 2 * w[j][i] + w[j - 1][i]) / (dy * dy * re) -
				jac;
		}
	}
}

void numericalDirect(double dx, double dy, double dt, v2d& wn, v2d& sn)
{
	static v2d r(nx + 2, v1d(ny + 2));

	rhs(dx, dy, wn, sn, r);
	for (int i = 1; i <= nx; ++i)
		for (int j = 1; j <= ny; ++j)
			wn[i][j] += dt * r[i][j];
	boundCond(wn);

	mgN(dx, dy, 5, wn, sn); boundCond(sn);
}

void initField(double dx, double dy, v2d& w) {
	double xc1 = PI - PI / 4, yc1 = PI;
	double xc2 = PI + PI / 4, yc2 = PI;

	for (int i = 1; i <= nx + 1; ++i) {
		for (int j = 1; j <= ny + 1; ++j) {
			w[i][j] =
				exp(-PI * (pow(i * dx - xc1, 2) + pow(j * dy - yc1, 2))) +
				exp(-PI * (pow(i * dx - xc2, 2) + pow(j * dy - yc2, 2)));
		}
	}
}

//[srcRa,srcRb]->[toRa,toRb]
constexpr auto valMap(double val, double srcRa = 0, double  srcRb = 0.1, double toRa = 0, double toRb = 240)
{
	double ratio = (srcRb - val) / (srcRb - srcRa);
	return toRb - ratio * (toRb - toRa);
}

void initGraph()
{
	SetWindowText(initgraph(nx, ny, 1), "Vortex Merger");
}

void display(const v2d& u)
{
	double minval = 99999, maxval = -99999;
	for (int i = 0; i < ny; i++)
	{
		minval = min(minval, *min_element(u[i].begin(), u[i].end()));
		maxval = max(maxval, *max_element(u[i].begin(), u[i].end()));
	}
	auto buf = GetImageBuffer();
	for (int i = 0; i < ny; i++)
		for (int j = 0; j < nx; j++)
			buf[i * nx + j] = HSVtoRGB(valMap(u[i][j], minval, maxval), 1, 1);
}

int main()
{
	initGraph();

	double dx = (2.0 * PI) / nx;
	double dy = (2.0 * PI) / ny;

	double dt = 0.001;
	int n = 1000000 / dt;

	v2d wn(ny + 2, v1d(nx + 2, 0));
	v2d sn(ny + 2, v1d(nx + 2, 0));
	v2d ln(ny + 2, v1d(nx + 2, 0));

	initField(dx, dy, wn); boundCond(wn);

	for (int t = 0; t < n; ++t) {

		numericalDirect(dx, dy, dt, wn, sn);
		if (t % 100 == 0)
		{
			display(wn);
		}
	}

	system("pause");
	return 0;
}