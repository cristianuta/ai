package ro.lab.ai.nn;

import java.util.List;
import java.util.Random;

/**
 * Rețea neurală complet conectată (MLP), scrisă de la zero.
 * <ul>
 *   <li>straturi ascunse: tanh</li>
 *   <li>ieșire: softmax + cross-entropy</li>
 *   <li>antrenare: backpropagation, mini-batch SGD cu momentum și regularizare L2</li>
 * </ul>
 * {@code weights[l]} are dimensiunea out×in și e indexat {@code weights[l][j * in + i]}
 * (conexiunea de la neuronul i din stratul l la neuronul j din stratul l+1).
 */
public class NeuralNetwork {

    public record Prediction(int label, double[] probabilities) {}

    public record Evaluation(double loss, double accuracy) {}

    private final int[] sizes;
    private final double[][] weights;
    private final double[][] biases;
    private final double[][] weightVelocity;
    private final double[][] biasVelocity;

    public NeuralNetwork(int[] sizes, Random rnd) {
        if (sizes.length < 2) {
            throw new IllegalArgumentException("Rețeaua are nevoie de cel puțin un strat de intrare și unul de ieșire");
        }
        this.sizes = sizes.clone();
        int layers = sizes.length - 1;
        weights = new double[layers][];
        biases = new double[layers][];
        weightVelocity = new double[layers][];
        biasVelocity = new double[layers][];
        for (int l = 0; l < layers; l++) {
            int nin = sizes[l], nout = sizes[l + 1];
            double scale = Math.sqrt(2.0 / (nin + nout)); // inițializare Xavier/Glorot
            weights[l] = new double[nin * nout];
            for (int k = 0; k < weights[l].length; k++) {
                weights[l][k] = rnd.nextGaussian() * scale;
            }
            biases[l] = new double[nout];
            weightVelocity[l] = new double[nin * nout];
            biasVelocity[l] = new double[nout];
        }
    }

    /** Activările tuturor straturilor: [intrare, ascuns1, ..., ieșire (probabilități)]. */
    public double[][] forward(double[] x) {
        if (x.length != sizes[0]) {
            throw new IllegalArgumentException("Intrarea trebuie să aibă " + sizes[0] + " valori");
        }
        int layers = weights.length;
        double[][] acts = new double[layers + 1][];
        acts[0] = x.clone();
        for (int l = 0; l < layers; l++) {
            int nin = sizes[l], nout = sizes[l + 1];
            double[] w = weights[l], b = biases[l], a = acts[l];
            double[] z = new double[nout];
            for (int j = 0; j < nout; j++) {
                double s = b[j];
                int off = j * nin;
                for (int i = 0; i < nin; i++) {
                    s += w[off + i] * a[i];
                }
                z[j] = s;
            }
            if (l == layers - 1) {
                softmaxInPlace(z);
            } else {
                for (int j = 0; j < nout; j++) {
                    z[j] = Math.tanh(z[j]);
                }
            }
            acts[l + 1] = z;
        }
        return acts;
    }

    private static void softmaxInPlace(double[] z) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : z) max = Math.max(max, v);
        double sum = 0;
        for (int j = 0; j < z.length; j++) {
            z[j] = Math.exp(z[j] - max);
            sum += z[j];
        }
        for (int j = 0; j < z.length; j++) z[j] /= sum;
    }

    public Prediction predict(double[] x) {
        double[] out = forward(x)[weights.length];
        int best = 0;
        for (int j = 1; j < out.length; j++) {
            if (out[j] > out[best]) best = j;
        }
        return new Prediction(best, out);
    }

    /** Un pas de antrenare pe un mini-batch. Returnează eroarea medie (cross-entropy). */
    public double trainBatch(double[][] xs, int[] ys, double learningRate, double momentum, double l2) {
        int layers = weights.length;
        double[][] gradW = new double[layers][];
        double[][] gradB = new double[layers][];
        for (int l = 0; l < layers; l++) {
            gradW[l] = new double[weights[l].length];
            gradB[l] = new double[biases[l].length];
        }

        double loss = 0;
        for (int s = 0; s < xs.length; s++) {
            double[][] acts = forward(xs[s]);
            double[] out = acts[layers];
            loss -= Math.log(Math.max(out[ys[s]], 1e-12));

            // gradientul softmax + cross-entropy față de z: p - y
            double[] delta = out.clone();
            delta[ys[s]] -= 1;

            for (int l = layers - 1; l >= 0; l--) {
                int nin = sizes[l], nout = sizes[l + 1];
                double[] a = acts[l], w = weights[l], gw = gradW[l], gb = gradB[l];
                double[] prev = l > 0 ? new double[nin] : null;
                for (int j = 0; j < nout; j++) {
                    double d = delta[j];
                    if (d == 0) continue;
                    gb[j] += d;
                    int off = j * nin;
                    for (int i = 0; i < nin; i++) {
                        gw[off + i] += d * a[i];
                        if (prev != null) prev[i] += w[off + i] * d;
                    }
                }
                if (prev != null) {
                    for (int i = 0; i < nin; i++) prev[i] *= 1 - a[i] * a[i]; // derivata tanh
                    delta = prev;
                }
            }
        }

        int n = xs.length;
        for (int l = 0; l < layers; l++) {
            double[] w = weights[l], vw = weightVelocity[l], gw = gradW[l];
            for (int k = 0; k < w.length; k++) {
                vw[k] = momentum * vw[k] - learningRate * (gw[k] / n + l2 * w[k]);
                w[k] += vw[k];
            }
            double[] b = biases[l], vb = biasVelocity[l], gb = gradB[l];
            for (int k = 0; k < b.length; k++) {
                vb[k] = momentum * vb[k] - learningRate * (gb[k] / n);
                b[k] += vb[k];
            }
        }
        return loss / n;
    }

    public Evaluation evaluate(List<double[]> xs, List<Integer> ys) {
        if (xs.isEmpty()) return new Evaluation(0, 0);
        double loss = 0;
        int correct = 0;
        for (int s = 0; s < xs.size(); s++) {
            Prediction p = predict(xs.get(s));
            int y = ys.get(s);
            loss -= Math.log(Math.max(p.probabilities()[y], 1e-12));
            if (p.label() == y) correct++;
        }
        return new Evaluation(loss / xs.size(), (double) correct / xs.size());
    }

    public int[] sizes() {
        return sizes.clone();
    }

    /** Acces direct (doar citire!) la ponderi — folosit pentru vizualizare. */
    double[][] weights() {
        return weights;
    }

    double[][] biases() {
        return biases;
    }

    public int parameterCount() {
        int count = 0;
        for (int l = 0; l < weights.length; l++) count += weights[l].length + biases[l].length;
        return count;
    }
}
