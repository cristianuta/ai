package ro.lab.ai.ga;

import java.util.Map;
import java.util.Random;

/** Ținte predefinite (8×8) pentru algoritmul genetic. */
public final class GaPresets {

    private static final Map<String, String[]> PRESETS = Map.of(
            "heart", new String[]{"........", ".##..##.", "########", "########", ".######.", "..####..", "...##...", "........"},
            "smile", new String[]{"..####..", ".#....#.", "#.#..#.#", "#......#", "#.#..#.#", "#..##..#", ".#....#.", "..####.."},
            "arrow", new String[]{"...#....", "..##....", ".#######", "########", ".#######", "..##....", "...#....", "........"},
            "eight", new String[]{"..####..", ".#....#.", ".#....#.", "..####..", ".#....#.", ".#....#.", ".#....#.", "..####.."},
            "checker", new String[]{"#.#.#.#.", ".#.#.#.#", "#.#.#.#.", ".#.#.#.#", "#.#.#.#.", ".#.#.#.#", "#.#.#.#.", ".#.#.#.#"}
    );

    private GaPresets() {
    }

    /** Ținta cu numele dat; „random” generează una aleatoare. */
    public static byte[] get(String name, Random rnd) {
        if ("random".equals(name)) {
            byte[] out = new byte[64];
            for (int i = 0; i < out.length; i++) out[i] = (byte) (rnd.nextBoolean() ? 1 : 0);
            return out;
        }
        String[] rows = PRESETS.get(name);
        if (rows == null) throw new IllegalArgumentException("Model necunoscut: " + name);
        byte[] out = new byte[64];
        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) out[y * 8 + x] = (byte) (rows[y].charAt(x) == '#' ? 1 : 0);
        }
        return out;
    }
}
