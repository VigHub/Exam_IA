# Esame di Intelligenza Artificiale 2019-2020 UNIBG
Il Paper studiato è **Batched Multi-armed Bandits Problem**, di 
Zijun Gao, Yanjun Han, Zhimei Ren, Zhengqing Zhou.  
Questa è la cartella contenente il codice del progetto sviluppato in Python.  
Sono presenti tre cartelle: 
+ **agents_simple_case**: codice relativo al capitolo 1 (Contesto), dove sono implementati 
tre diversi tipi di agenti che risolvono il problema dei banditi in uno scenario semplificato. 
Eseguire il file [main_agents.py](agents_simple_case/main_agents.py).
+ **paper_random_data**: codice relativo al capitolo 4 (Esperimenti), dove sono 
riproposte le stesse funzioni dello studio con dati casuali riscritte in Python.
Eseguire il file [main_random.py](paper_random_data/main_random.py).
+ **paper_real_data**: codice relativo al capitolo 4 (Esperimenti), dove sono 
presenti le funzioni dello studio con dati reali letti da un database.
Eseguire il file [main_real.py](paper_real_data/main_real.py).

Per maggiori dettagli e per leggere tutte le considerazioni si rimanda 
al PDF [TakeHome](TakeHome.pdf).

Pacchetti da installare con pip: 
+ numpy
+ pandas
+ matplotlib

