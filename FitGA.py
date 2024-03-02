import numpy as np
import math
import time
import skfuzzy as fuzz

import matplotlib.pyplot as plt


fig, ax = plt.subplots(2)
fig.set_size_inches(12,8)
fig.canvas.manager.window.move(1400,100)

# Definición de la función x^2 + x^3 + sen(x^2)
def function(x):
    return (x*x)-(x*x*x)+math.sin(x*x)

def plot_fmGaussiana():
    x = np.arange(-4, 4.001, 0.10)
    fp1 = fuzz.gaussmf(x, -2, 1)
    fp2 = fuzz.gaussmf(x, 2, 1)
    fig_scale = 2
    plt.figure(figsize=(6.4 * fig_scale, 4.8 * fig_scale))
    plt.subplot(111)
    plt.plot(x, fp1, label="fp1")
    plt.plot(x, fp2, label="fp2")

    plt.legend(loc="upper right")
    plt.ylabel("Membresía")
    plt.xlabel("Universo de Discurso")


def plot_2d(results: np.array, expected = [], labels=[''], title='', block=False):

    ax[0].clear()
    ax[0].plot(results.T[0], results.T[1], '-b', label=labels[0])
    if len(expected):
        ax[0].plot(expected.T[0], expected.T[1], '-r', label=labels[1])

    ax[0].set_ylabel('Y')
    ax[0].set_xlabel('X')
    ax[0].set_title(title)
    plt.show(block=block)
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.25)

def plot_2dError(results: np.array, expected = [], labels=[''], title='', block=False):

    ax[1].clear()

    ax[1].plot(results.T[0], results.T[1], '-b', label=labels[0])
    if len(expected):
        ax[1].plot(expected.T[0], expected.T[1], '-r', label=labels[1])

    ax[1].set_ylabel('Error')
    ax[1].set_xlabel('Epoca')
    ax[1].set_title(title)

    plt.show(block=block)
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.25)

def exp_cuadratica(x):
    return  math.exp(-1/2*math.pow(x,2))

def gaussian_mf(x, mean, std_dev):
    return np.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))

class FuzzyTsk:

    def shuffle_points(self):
        self.points_shuffled = np.random.permutation(self.points)

    def __init__(self, function, step=0.005, epocas=12500, min_x=-2, max_x=2, alpha=0.01):

        self.f = function
        self.step = step
        self.epocas = epocas
        self.min_x = min_x
        self.max_x = max_x
        self.alpha = alpha

        # generación de puntos
        self.points = np.array([(x, self.f(x)) for x in np.arange(-2,2,self.step)])
        self.shuffle_points()

    def estimar_valor_y(self, x1m, x1s, p1, q1, x2m, x2s, p2, q2, x, return_all=True):
        #Calcular w1 y w2 para calcular "y" de cada uno
        w1 = exp_cuadratica((x-x1m)/x1s)
        w2 = exp_cuadratica((x-x2m)/x2s)

        w1n = w1/(w1 + w2)
        w2n = w2/(w1 + w2)

        y1 = p1*x + q1
        y2 = p2*x + q2

        y = w1n*y1 + w2n*y2

        if return_all:
            return w1, w2, w1n, w2n, y1, y2, y
        return y

    def execute(self):

        x1m = -2
        x1s = 1
        p1 = -2
        q1 = 0

        x2m = 2
        x2s = 1
        p2 = 2
        q2 = 0

        plot_fmGaussiana()
        sumatoriaError = 0
        listaErrores = []
        for epoca in range(0, self.epocas):

            if not epoca or epoca % (self.epocas//150) == 0 or epoca == self.epocas - 1:
                puntos_estimados = np.array([(x, self.estimar_valor_y(x1m, x1s, p1,
                                                                q1, x2m, x2s,
                                                                p2, q2, x, return_all=False)) for x in np.arange(-2,2,self.step)])
                plot_2d(puntos_estimados, self.points, ['estimado', 'esperado'],'Despues de {} épocas - Error = {}'.format(epoca, round(sumatoriaError,2)))

            sumatoriaError = 0

            # Inferencia de los puntos generados
            for i,x in enumerate(self.points.T[0]):

                w1, w2, w1n, w2n, y1, y2, y = self.estimar_valor_y(x1m, x1s, p1, q1, x2m, x2s, p2, q2, x)
                yd = self.points[i][1]
                e = y - yd

                # calcular de los parametros
                p1 = p1 - self.alpha*e*w1n*x
                p2 = p2 - self.alpha*e*w2n*x

                q1 = q1 - self.alpha*e*w1n
                q2 = q2 - self.alpha*e*w2n

                x1m = x1m - self.alpha*e*w2*(y1-y2)/(pow(w1+w2, 2))*(x-x1m)/pow(x1s,2)*exp_cuadratica((x-x1m)/x1s)
                x2m = x2m - self.alpha*e*w1*(y2-y1)/(pow(w1+w2, 2))*(x-x2m)/pow(x2s,2)*exp_cuadratica((x-x2m)/x2s)

                x1s = x1s - self.alpha*e*w2*(y1-y2)/(pow(w1+w2, 2))*pow(x-x1m,2)/pow(x1s,3)*exp_cuadratica((x-x1m)/x1s)
                x2s = x2s - self.alpha*e*w1*(y2-y1)/(pow(w1+w2, 2))*pow(x-x2m,2)/pow(x2s,3)*exp_cuadratica((x-x2m)/x2s)

                sumatoriaError = sumatoriaError + pow(e, 2)

            # generar nuevos puntos para la nueva iteración
            self.shuffle_points()

            listaErrores.append((int(epoca), sumatoriaError))
            plot_2dError(np.array(listaErrores), labels=['Error'], title='Error por epoca', block=False)

if __name__ == "__main__":

    ft = FuzzyTsk(function, alpha=0.01, epocas=150)
    ft.execute()
    time.sleep(15)