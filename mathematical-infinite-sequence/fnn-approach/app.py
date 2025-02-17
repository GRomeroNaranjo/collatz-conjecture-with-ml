import pandas as pd
import os

def collatz_steps(n):
    steps = 0
    while n != 1:
        if n % 2 == 0:
            n //= 2
        else:
            n = 3 * n + 1
        steps += 1
    return n, steps

numbers = []
steps_list = []

num = 1

while num <= 50000000:
    n, total_steps = collatz_steps(num)
    print(f"Number: {num}, Final Number: {n} Steps: {total_steps}")

    numbers.append(num)
    steps_list.append(total_steps)

    num += 1

df_numbers = pd.DataFrame({"Number": numbers})
df_steps = pd.DataFrame({"Steps": steps_list})

folder_path = "/Users/gromeronaranjo/Desktop/mathematical-infinite-sequence/"
os.makedirs(folder_path, exist_ok=True)

df_numbers.to_csv(os.path.join(folder_path, "numbers.csv"), index=False)
df_steps.to_csv(os.path.join(folder_path, "steps.csv"), index=False)

