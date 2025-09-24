import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

df = pd.read_csv("food.csv")
category_map = {
    # Молочные продукты
    "BUTTER": "Масло сливочное, кг",
    "CHEESE": "Национальные сыры и брынза, кг",
    "CHEESE FD": "Сыры плавленые, кг",
    "CHEESE SPRD": "Сыры плавленые, кг",
    "MILK": "Молоко питьевое цельное пастеризованное 2,5-3,2% жирности, л",
    "MILK SHAKES": "Молоко питьевое цельное пастеризованное 2,5-3,2% жирности, л",
    "YOGURT": "Кисломолочные продукты, кг",
    "EGG": "Яйца куриные, 10 шт.",

    # Овощи
    "CARROT": "Морковь, кг",
    "TOMATO": "Помидоры свежие, кг",
    "CUCUMBER": "Огурцы свежие, кг",
    "POTATO": "Картофель, кг",
    "ONION": "Лук репчатый, кг",
    "GARLIC": "Чеснок, кг",
    "CABBAGE": "Капуста белокочанная свежая, кг",

    # Фрукты
    "APPLES": "Яблоки, кг",
    "BANANAS": "Бананы, кг",
    "ORANGES": "Апельсины, кг",
    "LEMONS": "Лимоны, кг",
    "PEARS": "Груши, кг",

    # Мясо и птица
    "CHICKEN": "Куры охлажденные и мороженые, кг",
    "TURKEY": "Мясо индейки, кг",
    "PORK": "Свинина бескостная, кг",
    "BEEF": "Говядина бескостная, кг",

    # Рыба и морепродукты
    "FISH": "Филе рыбное, кг",
    "SALMON": "Рыба мороженая разделанная (кроме лососевых пород), кг",
    "SHRIMP": "Креветки мороженые неразделанные, кг",

    # Зерновые и крупы
    "RICE": "Рис шлифованный, кг",
    "WHEAT": "Мука пшеничная, кг",
    "OAT": "Хлопья из злаков (сухие завтраки), кг",

    # Сладости и напитки
    "COFFEE": "Кофе натуральный растворимый, кг",
    "TEA": "Чай черный байховый, кг",
}

df_small = df[df['Category'].isin(category_map.keys())].copy()
df_small = df_small[
    ['Category', 'Description', 'Data.Kilocalories', 'Data.Carbohydrate', 'Data.Fat.Total Lipid', 'Data.Protein']]

nutrient_cols = ['Data.Kilocalories', 'Data.Carbohydrate', 'Data.Fat.Total Lipid', 'Data.Protein']
df_small[nutrient_cols] = df_small[nutrient_cols] * 10
df_small[nutrient_cols] = df_small[nutrient_cols].round(1)

df_small['local_product_name'] = df_small['Category'].map(category_map)

df_prices = pd.read_excel("price.xls", header=None)
df_prices = df_prices.iloc[:, [0, 1]]
df_prices.columns = ["local_product_name", "price"]
df_prices["local_product_name"] = df_prices["local_product_name"].str.strip()
df_prices["price"] = df_prices["price"].astype(str).str.replace(" ", "").str.replace(",", ".").astype(float)

df_final = df_small.merge(df_prices, on="local_product_name", how="left")

# НОРМА
calories_norm = 2150 * 7#ккал
protein_norm = 75 * 7#белки
fat_norm = 72 * 7#жиры
carbs_norm = 301 * 7#углеводы

N = len(df_final)
POP_SIZE = 100
GENERATIONS = 200
K = 10

# Функция приспособленности
def fitness(chromosome):
    # Проверяем количество выбранных продуктов
    selected_indices = np.where(chromosome == 1)[0]
    num_selected = len(selected_indices)

    # Большой штраф за неправильное количество продуктов
    if num_selected != K:
        return float('inf')

    selected = df_final.iloc[selected_indices]

    total_calories = selected['Data.Kilocalories'].sum()
    total_protein = selected['Data.Protein'].sum()
    total_fat = selected['Data.Fat.Total Lipid'].sum()
    total_carbs = selected['Data.Carbohydrate'].sum()
    total_price = selected['price'].sum()

    # Относительные отклонения от норм
    calorie_dev = abs(total_calories - calories_norm) / calories_norm
    protein_dev = abs(total_protein - protein_norm) / protein_norm
    fat_dev = abs(total_fat - fat_norm) / fat_norm
    carbs_dev = abs(total_carbs - carbs_norm) / carbs_norm

    penalty_scale = 10000

    calorie_penalty = calorie_dev * penalty_scale
    protein_penalty = protein_dev * penalty_scale
    fat_penalty = fat_dev * penalty_scale
    carbs_penalty = carbs_dev * penalty_scale

    total_penalty = calorie_penalty + protein_penalty + fat_penalty + carbs_penalty

    return total_price + total_penalty

# Функция восстановления хромосомы
def repair_chromosome(chromosome):
    ones_indices = np.where(chromosome == 1)[0]
    num_ones = len(ones_indices)

    if num_ones == K:
        return chromosome
    elif num_ones > K:
        to_remove = np.random.choice(ones_indices, size=num_ones - K, replace=False)
        chromosome[to_remove] = 0
    else:
        zeros_indices = np.where(chromosome == 0)[0]
        to_add = np.random.choice(zeros_indices, size=K - num_ones, replace=False)
        chromosome[to_add] = 1

    return chromosome

# Скрещивание
def one_point_crossover(p1, p2):
    point = random.randint(1, N - 1)
    c1 = np.concatenate([p1[:point], p2[point:]])
    c2 = np.concatenate([p2[:point], p1[point:]])
    return repair_chromosome(c1), repair_chromosome(c2)


def two_point_crossover(p1, p2):
    points = sorted(random.sample(range(1, N), 2))
    c1 = np.concatenate([p1[:points[0]], p2[points[0]:points[1]], p1[points[1]:]])
    c2 = np.concatenate([p2[:points[0]], p1[points[0]:points[1]], p2[points[1]:]])
    return repair_chromosome(c1), repair_chromosome(c2)


def uniform_crossover(p1, p2):
    mask = np.random.randint(0, 2, size=N)
    c1 = np.where(mask, p1, p2)
    c2 = np.where(mask, p2, p1)
    return repair_chromosome(c1), repair_chromosome(c2)

def mutation_single_gene(chromosome, prob=0.1):
    chromosome = chromosome.copy()

    if random.random() < prob:
        gene_index = random.randint(0, N - 1)
        chromosome[gene_index] = 1 - chromosome[gene_index]

    return repair_chromosome(chromosome)


def mutation_multiple_genes(chromosome, prob=0.1):
    chromosome = chromosome.copy()

    if random.random() < prob:
        # Меняем от 1 до 3 случайных генов
        num_mutations = random.randint(1, 3)
        for _ in range(num_mutations):
            gene_index = random.randint(0, N - 1)
            chromosome[gene_index] = 1 - chromosome[gene_index]

    return repair_chromosome(chromosome)


def mutation_improvement(chromosome, prob=0.1):
    """Улучшающая мутация - целенаправленная замена дорогого продукта"""
    chromosome = chromosome.copy()
    ones_indices = np.where(chromosome == 1)[0]

    if random.random() < prob and len(ones_indices) > 0:
        # Находим самый дорогой продукт в рационе
        selected = df_final.iloc[ones_indices]
        expensive_idx = selected['price'].idxmax()

        # Ищем дешевую замену из той же категории
        expensive_product = df_final.loc[expensive_idx]
        same_category = df_final[
            (df_final['Category'] == expensive_product['Category']) &
            (df_final.index != expensive_idx)
            ]

        if not same_category.empty:
            # Берем самый дешевый продукт из той же категории
            cheap_replacement = same_category['price'].idxmin()
            chromosome[expensive_idx] = 0
            chromosome[cheap_replacement] = 1

    return chromosome


# Создание популяции и отбор
def create_population():
    population = []
    for _ in range(POP_SIZE):
        chromosome = np.zeros(N, dtype=int)
        indices = np.random.choice(N, size=K, replace=False)
        chromosome[indices] = 1
        population.append(chromosome)
    return np.array(population)


def tournament_selection(pop, k=3):
    selected = []
    for _ in range(len(pop)):
        aspirants = random.sample(list(pop), k)
        f_values = [fitness(a) for a in aspirants]
        selected.append(aspirants[np.argmin(f_values)])
    return np.array(selected)


# Эксперименты с разными методами
def run_experiment(crossover_func, mutation_func, experiment_name):
    pop = create_population()
    best_fit_history = []

    for gen in range(GENERATIONS):
        # Отбор
        pop = tournament_selection(pop)
        next_pop = []

        # Скрещивание
        for i in range(0, POP_SIZE, 2):
            if i + 1 < len(pop):
                p1, p2 = pop[i], pop[i + 1]
                c1, c2 = crossover_func(p1, p2)
                next_pop.extend([c1, c2])

        # Мутация
        for i in range(len(next_pop)):
            next_pop[i] = mutation_func(next_pop[i])

        pop = np.array(next_pop)
        fitness_values = [fitness(c) for c in pop]
        best_fit_history.append(min(fitness_values))

        if gen % 40 == 0:
            print(f"{experiment_name} - Поколение {gen}: {min(fitness_values):.2f}")

    # Анализ лучшего решения
    best_idx = np.argmin([fitness(c) for c in pop])
    best_solution = pop[best_idx]
    selected = df_final.iloc[np.where(best_solution == 1)[0]]

    total_calories = selected['Data.Kilocalories'].sum()
    total_protein = selected['Data.Protein'].sum()
    total_fat = selected['Data.Fat.Total Lipid'].sum()
    total_carbs = selected['Data.Carbohydrate'].sum()
    total_price = selected['price'].sum()

    print(f"\n=== {experiment_name} ===")
    print(f"Калории: {total_calories:.1f}/{calories_norm} ({total_calories / calories_norm * 100:.1f}%)")
    print(f"Белки: {total_protein:.1f}/{protein_norm}г ({total_protein / protein_norm * 100:.1f}%)")
    print(f"Жиры: {total_fat:.1f}/{fat_norm}г ({total_fat / fat_norm * 100:.1f}%)")
    print(f"Углеводы: {total_carbs:.1f}/{carbs_norm}г ({total_carbs / carbs_norm * 100:.1f}%)")
    print(f"Стоимость: {total_price:.2f} руб.")
    print("Продукты:")
    for _, row in selected.iterrows():
        print(f"  - {row['local_product_name']}")

    return best_fit_history


print("=== СРАВНЕНИЕ МЕТОДОВ СКРЕЩИВАНИЯ ===")
crossovers = [
    (one_point_crossover, "Одноточечное скрещивание"),
    (two_point_crossover, "Двухточечное скрещивание"),
    (uniform_crossover, "Равномерное скрещивание")
]

plt.figure(figsize=(12, 6))
for crossover, name in crossovers:
    history = run_experiment(crossover, mutation_single_gene, name)
    plt.plot(history, label=name)

plt.xlabel("Поколение")
plt.ylabel("Функция приспособленности")
plt.title("Сравнение методов скрещивания")
plt.legend()
plt.grid(True)
plt.show()

print("\n=== СРАВНЕНИЕ МЕТОДОВ МУТАЦИИ ===")
mutations = [
    (mutation_single_gene, "Точечная мутация"),
    (mutation_multiple_genes, "Множественная мутация"),
    (mutation_improvement, "Улучшающая мутация")
]

plt.figure(figsize=(12, 6))
for mutation, name in mutations:
    history = run_experiment(one_point_crossover, mutation, name)
    plt.plot(history, label=name)

plt.xlabel("Поколение")
plt.ylabel("Функция приспособленности")
plt.title("Сравнение методов мутации")
plt.legend()
plt.grid(True)
plt.show()