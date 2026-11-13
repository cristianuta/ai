package ro.lab.ai;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Laborator AI: o rețea neurală 8×8 și un algoritm genetic.
 * Toate calculele rulează în Java; front-end-ul (src/main/resources/static) doar le desenează.
 */
@SpringBootApplication
public class AiLabApplication {

    public static void main(String[] args) {
        SpringApplication.run(AiLabApplication.class, args);
    }
}
