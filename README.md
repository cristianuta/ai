# ai — Laborator AI
Algoritmi genetici și rețele neurale: demonstrații interactive cu back-end **Java (Spring Boot 4, Maven)**
și front-end în browser.

Toate calculele rulează în Java pe server: antrenarea rețelei (backpropagation) și evoluția populației.
Browserul doar desenează rezultatele primite prin API-ul REST. Starea (rețeaua, populația) e păstrată
**per sesiune HTTP**, deci fiecare utilizator are propriul laborator.

## Pornire
Necesită JDK 21+. Maven nu trebuie instalat — proiectul include Maven Wrapper.

```bash
./mvnw spring-boot:run        # Windows: mvnw.cmd spring-boot:run
```
Apoi deschide <http://localhost:8080>.

Alte comenzi:
```bash
./mvnw test                   # testele unitare
./mvnw package                # construiește target/ai-lab-1.0.0.jar
java -jar target/ai-lab-1.0.0.jar
```
Din IntelliJ: rulează clasa `ro.lab.ai.AiLabApplication`.

## Tab 1 · Rețea neurală 8×8
- Desenezi o cifră pe o grilă de 8×8 pixeli → 64 de neuroni de intrare.
- MLP scris de la zero (`NeuralNetwork.java`): straturi ascunse configurabile (tanh), 10 ieșiri (softmax),
  backpropagation cu mini-batch SGD + momentum + L2.
- Nodurile și conexiunile sunt desenate live (ponderi sau semnalul pentru desenul curent);
  hover pe un neuron îi evidențiază conexiunile.
- Grafice de eroare și acuratețe (antrenare vs. test) și „ce vede” fiecare neuron din stratul 1.
- Poți adăuga propriile desene în setul de antrenare.

## Tab 2 · Algoritm genetic
- Indivizi = vectori de 64 de biți (imagini 8×8); fitness = câte gene se potrivesc cu ținta.
- Selecție turneu, încrucișare (1 punct / 2 puncte / uniformă), mutație, elitism (`GeneticAlgorithm.java`).
- Vizualizarea încrucișării: doi vectori părinte → copil, cu bucata moștenită de la fiecare și genele „bune”.
- Fondul genetic arată cum genele corecte se răspândesc în populație de la o generație la alta.

## API REST
| Metodă | Cale | Descriere |
|---|---|---|
| POST | `/api/nn/state` | starea rețelei (+ activările pentru `input`) |
| POST | `/api/nn/predict` | doar activările pentru un desen |
| POST | `/api/nn/train` | `batches` mini-batch-uri de antrenare |
| POST | `/api/nn/epoch` | o epocă completă |
| POST | `/api/nn/reset` | rețea nouă cu straturile `hidden` |
| POST | `/api/nn/data` | date noi cu nivelul de zgomot `noise` |
| POST | `/api/nn/samples` | adaugă un desen etichetat la antrenare |
| GET | `/api/nn/random-sample` | un exemplu aleator din setul de test |
| GET | `/api/ga/state` | starea algoritmului genetic |
| POST | `/api/ga/reset` | populație nouă (mărime, mutație, încrucișare, elitism) |
| POST | `/api/ga/step` | rulează `generations` generații |
| POST | `/api/ga/target` | schimbă ținta (`target` = 64 de biți sau `preset`) |

## Structură
```
pom.xml                                   Maven (Spring Boot 4.1, Java 21)
src/main/java/ro/lab/ai/
  AiLabApplication.java                   punctul de pornire
  nn/NeuralNetwork.java                   rețeaua neurală (forward, backprop)
  nn/DigitDataset.java                    șabloane de cifre + generare date
  nn/NeuralNetworkService.java            starea per sesiune
  nn/NeuralNetworkController.java         API /api/nn
  ga/GeneticAlgorithm.java                algoritmul genetic
  ga/GaPresets.java                       ținte predefinite
  ga/GeneticAlgorithmService.java         starea per sesiune
  ga/GeneticAlgorithmController.java      API /api/ga
  web/ApiExceptionHandler.java            erori JSON
src/main/resources/static/                front-end (HTML, CSS, JS — doar vizualizare)
src/test/java/...                         teste: gradient numeric, învățare, convergența GA
```
