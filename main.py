import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random

class TrafficSignExpertSystem:
    def __init__(self):
        self.rules = [
            {"condition": "color == 'red' and shape == 'circle'", "action": "sign_type = 'prohibitory'"},
            {"condition": "color == 'red' and shape == 'triangle'", "action": "sign_type = 'warning'"},
            {"condition": "color == 'blue' and shape == 'rectangle'", "action": "sign_type = 'information'"},
            {"condition": "shape == 'octagon'", "action": "sign_type = 'stop_sign'"},
            {"condition": "color == 'red' and shape == 'octagon'", "action": "sign_name = 'STOP'"},
            {"condition": "color == 'red' and shape == 'circle' and has_number == True", "action": "sign_name = 'SPEED_LIMIT'"}
        ]
    
    def evaluate_rule(self, condition, facts):
        try:
            return eval(condition, {}, facts)  # Використовуємо порожній словник для builtins
        except:
            return False
    
    def infer(self, facts):
        local_facts = facts.copy()  # Працюємо з копією фактів
        for rule in self.rules:
            if self.evaluate_rule(rule["condition"], local_facts):
                # Виконуємо дію без додавання builtins
                if "sign_type" in rule["action"]:
                    local_facts["sign_type"] = rule["action"].split("= ")[1].strip("'")
                elif "sign_name" in rule["action"]:
                    local_facts["sign_name"] = rule["action"].split("= ")[1].strip("'")
        return local_facts

class SemanticNetwork:
    def __init__(self):
        self.relations = []
    
    def add_relation(self, from_node, relation_type, to_node):
        self.relations.append({'from': from_node, 'relation': relation_type, 'to': to_node})
    
    def query(self, start_node, relation_type=None):
        return [rel for rel in self.relations if rel['from'] == start_node and (relation_type is None or rel['relation'] == relation_type)]

class Frame:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.slots = {}
    
    def set_slot(self, slot_name, value):
        self.slots[slot_name] = value
    
    def get_slot(self, slot_name):
        if slot_name in self.slots:
            return self.slots[slot_name]
        elif self.parent:
            return self.parent.get_slot(slot_name)
        return None

class TrafficSignNeuralNetwork:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def generate_data(self, n_samples=1000):
        np.random.seed(42)
        # Характеристики: [червоний_колір, синій_колір, круг_форма, трикутник_форма, наявність_тексту]
        X = np.random.randn(n_samples, 5)
        y = np.zeros(n_samples)
        for i in range(n_samples):
            if X[i, 0] > 0.5 and X[i, 2] > 0:  # червоний + круг
                y[i] = 0  # заборонний
            elif X[i, 0] > 0.5 and X[i, 3] > 0:  # червоний + трикутник
                y[i] = 1  # попереджувальний
            elif X[i, 1] > 0.5:  # синій
                y[i] = 2  # інформаційний
            else:
                y[i] = random.choice([0, 1, 2])
        return X, y
    
    def train(self):
        X, y = self.generate_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42, alpha=0.01)
        self.model.fit(X_train_scaled, y_train)
        return self.model.score(X_train_scaled, y_train), self.model.score(X_test_scaled, y_test)
    
    def predict(self, features):
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        class_names = ['Заборонний', 'Попереджувальний', 'Інформаційний']
        return {
            'class': class_names[int(prediction)],
            'probabilities': {class_names[i]: f"{prob:.3f}" for i, prob in enumerate(probabilities)}
        }

# Тестування систем
print("=== ТЕСТУВАННЯ СИСТЕМ РОЗПІЗНАВАННЯ ДОРОЖНІХ ЗНАКІВ ===\n")

# 1. Експертна система
print("1. ЕКСПЕРТНА СИСТЕМА (продукційні правила):")
print("--------------------------------------------")
expert = TrafficSignExpertSystem()

# Тест 1: Стоп-знак
test_facts_1 = {'color': 'red', 'shape': 'octagon', 'has_number': False}
result_1 = expert.infer(test_facts_1)
print("Тест 1 - Стоп-знак:")
print(f"   Вхідні параметри: колір={test_facts_1['color']}, форма={test_facts_1['shape']}")
print(f"   Результат: {result_1['sign_type']} - {result_1['sign_name']}")

# Тест 2: Знак обмеження швидкості
test_facts_2 = {'color': 'red', 'shape': 'circle', 'has_number': True}
result_2 = expert.infer(test_facts_2)
print("\nТест 2 - Знак обмеження швидкості:")
print(f"   Вхідні параметри: колір={test_facts_2['color']}, форма={test_facts_2['shape']}, має_число={test_facts_2['has_number']}")
print(f"   Результат: {result_2['sign_type']} - {result_2['sign_name']}")

# Тест 3: Попереджувальний знак
test_facts_3 = {'color': 'red', 'shape': 'triangle', 'has_number': False}
result_3 = expert.infer(test_facts_3)
print("\nТест 3 - Попереджувальний знак:")
print(f"   Вхідні параметри: колір={test_facts_3['color']}, форма={test_facts_3['shape']}")
print(f"   Результат: {result_3['sign_type']}")

# 2. Семантична мережа
print("\n2. СЕМАНТИЧНА МЕРЕЖА:")
print("---------------------")
semantic_net = SemanticNetwork()
relations = [
    ('stop_sign', 'is_a', 'prohibitory'),
    ('speed_limit', 'is_a', 'prohibitory'),
    ('yield_sign', 'is_a', 'warning'),
    ('information_sign', 'is_a', 'information'),
    ('prohibitory', 'part_of', 'traffic_rules'),
    ('warning', 'part_of', 'traffic_rules'),
    ('stop_sign', 'has_color', 'red'),
    ('speed_limit', 'has_shape', 'circle'),
    ('yield_sign', 'has_shape', 'triangle')
]

for rel in relations:
    semantic_net.add_relation(*rel)

print("Запит: Всі зв'язки для 'stop_sign':")
results = semantic_net.query('stop_sign')
for rel in results:
    print(f"   {rel['from']} --{rel['relation']}--> {rel['to']}")

print("\nЗапит: Всі знаки типу 'prohibitory':")
prohibitory_signs = semantic_net.query('prohibitory', 'is_a')
for rel in prohibitory_signs:
    print(f"   {rel['to']} --{rel['relation']}--> {rel['from']}")

# 3. Фреймова система
print("\n3. ФРЕЙМОВА СИСТЕМА:")
print("-------------------")

# Створення ієрархії фреймів
traffic_sign = Frame("TrafficSign")
traffic_sign.set_slot("призначення", "регулювання руху")
traffic_sign.set_slot("розташування", "біля дороги")

prohibitory_sign = Frame("ProhibitorySign", traffic_sign)
prohibitory_sign.set_slot("колір", "червоний")
prohibitory_sign.set_slot("значення", "заборона")

warning_sign = Frame("WarningSign", traffic_sign)
warning_sign.set_slot("колір", "червоний")
warning_sign.set_slot("форма", "трикутник")
warning_sign.set_slot("значення", "попередження")

stop_sign = Frame("StopSign", prohibitory_sign)
stop_sign.set_slot("форма", "восьмикутник")
stop_sign.set_slot("дія", "зупинити транспорт")

speed_limit_sign = Frame("SpeedLimitSign", prohibitory_sign)
speed_limit_sign.set_slot("форма", "круг")
speed_limit_sign.set_slot("має_число", True)

print("Тест 1 - Стоп-знак (фрейм StopSign):")
print(f"   Призначення: {stop_sign.get_slot('призначення')}")
print(f"   Колір: {stop_sign.get_slot('колір')}")
print(f"   Форма: {stop_sign.get_slot('форма')}")
print(f"   Дія: {stop_sign.get_slot('дія')}")
print(f"   Розташування: {stop_sign.get_slot('розташування')}")

print("\nТест 2 - Знак обмеження швидкості (фрейм SpeedLimitSign):")
print(f"   Призначення: {speed_limit_sign.get_slot('призначення')}")
print(f"   Колір: {speed_limit_sign.get_slot('колір')}")
print(f"   Форма: {speed_limit_sign.get_slot('форма')}")
print(f"   Має число: {speed_limit_sign.get_slot('має_число')}")

# 4. Нейронна мережа
print("\n4. НЕЙРОННА МЕРЕЖА:")
print("-------------------")
nn = TrafficSignNeuralNetwork()
print("Навчання нейронної мережі...")
train_acc, test_acc = nn.train()
print(f"   Точність на навчальних даних: {train_acc:.3f}")
print(f"   Точність на тестових даних: {test_acc:.3f}")

# Тестові приклади
test_cases = [
    [1.0, 0.0, 1.0, 0.0, 0.0],  # Заборонний знак (червоний, круг)
    [1.0, 0.0, 0.0, 1.0, 0.0],  # Попереджувальний знак (червоний, трикутник)
    [0.0, 1.0, 0.0, 0.0, 1.0],  # Інформаційний знак (синій, з текстом)
    [0.5, 0.5, 0.3, 0.3, 0.5]   # Невизначений знак
]

print("\nТестові передбачення:")
class_descriptions = {
    'Заборонний': 'знаки, що забороняють певні дії',
    'Попереджувальний': 'знаки, що попереджають про небезпеку',
    'Інформаційний': 'знаки, що надають інформацію'
}

for i, features in enumerate(test_cases, 1):
    result = nn.predict(features)
    print(f"   Тест {i}: {result['class']}")
    print(f"      Опис: {class_descriptions[result['class']]}")
    print(f"      Ймовірності: {result['probabilities']}")

print("\n=== ТЕСТУВАННЯ ЗАВЕРШЕНО ===")
print("Усі системи успішно протестовано!")