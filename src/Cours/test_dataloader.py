from Cours.dataloader import DataLoader 
import time

loader = DataLoader(cache_dir=".cache")

# Premier call
start = time.perf_counter()
ts1 = loader.fetch_single_ticker("AAPL", "Close", ("2024-01-01", "2024-06-01"))
premier_temps = time.perf_counter() - start
print(f"Premier fetch: {premier_temps:.2f} secondes")

# deuxieme call depuis le cache
start = time.perf_counter()
ts2 = loader.fetch_single_ticker("AAPL", "Close", ("2024-01-01", "2024-06-01"))
second_temps = time.perf_counter() - start
print(f"Second fetch: {second_temps:.4f} secondes")

print(f"Accélération: {premier_temps/second_temps:.0f}x plus rapide avec le cache")