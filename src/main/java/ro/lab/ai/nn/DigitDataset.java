package ro.lab.ai.nn;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/** Șabloane de cifre 8×8 și generarea setului de date (cu deplasări și zgomot). */
public final class DigitDataset {

    public static final int SIDE = 8;
    public static final int PIXELS = SIDE * SIDE;

    /** Setul de exemple: imaginile (64 de valori 0..1) și etichetele lor (0..9). Listele sunt modificabile. */
    public record Data(List<double[]> xs, List<Integer> ys) {
        public int size() {
            return xs.size();
        }
    }

    private static final String[][] TEMPLATES = {
        {"..####..", ".#....#.", ".#....#.", ".#....#.", ".#....#.", ".#....#.", ".#....#.", "..####.."},
        {"...##...", "..###...", ".#.##...", "...##...", "...##...", "...##...", "...##...", "..####.."},
        {"..####..", ".#....#.", "......#.", ".....#..", "....#...", "...#....", "..#.....", ".######."},
        {"..####..", ".#....#.", "......#.", "...###..", "......#.", "......#.", ".#....#.", "..####.."},
        {"....##..", "...#.#..", "..#..#..", ".#...#..", ".######.", ".....#..", ".....#..", ".....#.."},
        {".######.", ".#......", ".#......", ".#####..", "......#.", "......#.", ".#....#.", "..####.."},
        {"..####..", ".#......", ".#......", ".#####..", ".#....#.", ".#....#.", ".#....#.", "..####.."},
        {".######.", "......#.", ".....#..", ".....#..", "....#...", "....#...", "...#....", "...#...."},
        {"..####..", ".#....#.", ".#....#.", "..####..", ".#....#.", ".#....#.", ".#....#.", "..####.."},
        {"..####..", ".#....#.", ".#....#.", ".#....#.", "..#####.", "......#.", ".....#..", "..###..."},
    };

    private DigitDataset() {
    }

    /** Transformă un desen text ('#' = aprins) în 64 de valori. */
    public static double[] pattern(String... rows) {
        double[] out = new double[PIXELS];
        for (int y = 0; y < SIDE; y++) {
            for (int x = 0; x < SIDE; x++) {
                out[y * SIDE + x] = rows[y].charAt(x) == '#' ? 1 : 0;
            }
        }
        return out;
    }

    public static double[] template(int digit) {
        return pattern(TEMPLATES[digit]);
    }

    /** O variație a imaginii: deplasare cu ±1 pixel, pixeli stinși/aprinși aleator, intensități variate. */
    public static double[] augment(double[] base, double noise, Random rnd) {
        int dx = rnd.nextInt(3) - 1;
        int dy = rnd.nextInt(3) - 1;
        double[] out = new double[PIXELS];
        for (int y = 0; y < SIDE; y++) {
            for (int x = 0; x < SIDE; x++) {
                int sx = x - dx, sy = y - dy;
                double v = (sx >= 0 && sx < SIDE && sy >= 0 && sy < SIDE) ? base[sy * SIDE + sx] : 0;
                if (v > 0) {
                    v = rnd.nextDouble() < noise * 0.6 ? 0 : 1 - rnd.nextDouble() * noise;
                } else if (rnd.nextDouble() < noise * 0.3) {
                    v = 0.4 + rnd.nextDouble() * 0.6;
                }
                out[y * SIDE + x] = v;
            }
        }
        return out;
    }

    public static Data generate(int perClass, double noise, Random rnd) {
        List<double[]> xs = new ArrayList<>();
        List<Integer> ys = new ArrayList<>();
        for (int d = 0; d < 10; d++) {
            double[] base = template(d);
            for (int k = 0; k < perClass; k++) {
                xs.add(augment(base, noise, rnd));
                ys.add(d);
            }
        }
        return new Data(xs, ys);
    }
}
