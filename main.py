import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import skfuzzy as fuzz
from skfuzzy import control as ctrl

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
            return eval(condition, {}, facts)
        except:
            return False
    
    def infer(self, facts):
        local_facts = facts.copy()
        for rule in self.rules:
            if self.evaluate_rule(rule["condition"], local_facts):
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
        X = np.random.randn(n_samples, 5)
        y = np.zeros(n_samples)
        for i in range(n_samples):
            if X[i, 0] > 0.5 and X[i, 2] > 0:
                y[i] = 0
            elif X[i, 0] > 0.5 and X[i, 3] > 0:
                y[i] = 1
            elif X[i, 1] > 0.5:
                y[i] = 2
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

class FuzzyTrafficSignSystem:
    def __init__(self):
        # Створюємо нечіткі змінні
        self.color_red = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'color_red')
        self.shape_circle = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'shape_circle')
        self.shape_triangle = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'shape_triangle')
        self.sign_type = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'sign_type')
        
        # Визначаємо функції належності
        self.color_red['low'] = fuzz.trimf(self.color_red.universe, [0, 0, 0.5])
        self.color_red['medium'] = fuzz.trimf(self.color_red.universe, [0, 0.5, 1])
        self.color_red['high'] = fuzz.trimf(self.color_red.universe, [0.5, 1, 1])
        
        self.shape_circle['low'] = fuzz.trimf(self.shape_circle.universe, [0, 0, 0.5])
        self.shape_circle['medium'] = fuzz.trimf(self.shape_circle.universe, [0, 0.5, 1])
        self.shape_circle['high'] = fuzz.trimf(self.shape_circle.universe, [0.5, 1, 1])
        
        self.shape_triangle['low'] = fuzz.trimf(self.shape_triangle.universe, [0, 0, 0.5])
        self.shape_triangle['medium'] = fuzz.trimf(self.shape_triangle.universe, [0, 0.5, 1])
        self.shape_triangle['high'] = fuzz.trimf(self.shape_triangle.universe, [0.5, 1, 1])
        
        self.sign_type['prohibitory'] = fuzz.trimf(self.sign_type.universe, [0, 0, 0.5])
        self.sign_type['warning'] = fuzz.trimf(self.sign_type.universe, [0, 0.5, 1])
        self.sign_type['information'] = fuzz.trimf(self.sign_type.universe, [0.5, 1, 1])
        
        # Створюємо правила
        rule1 = ctrl.Rule(self.color_red['high'] & self.shape_circle['high'], self.sign_type['prohibitory'])
        rule2 = ctrl.Rule(self.color_red['high'] & self.shape_triangle['high'], self.sign_type['warning'])
        rule3 = ctrl.Rule(self.color_red['low'] & self.shape_circle['medium'], self.sign_type['information'])
        rule4 = ctrl.Rule(self.color_red['medium'] & self.shape_triangle['medium'], self.sign_type['warning'])
        
        # Створюємо систему керування
        self.sign_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
        self.sign_system = ctrl.ControlSystemSimulation(self.sign_ctrl)
    
    def classify(self, color_red, shape_circle, shape_triangle):
        self.sign_system.input['color_red'] = color_red
        self.sign_system.input['shape_circle'] = shape_circle
        self.sign_system.input['shape_triangle'] = shape_triangle
        
        try:
            self.sign_system.compute()
            result = self.sign_system.output['sign_type']
            
            if result <= 0.4:
                return 'Заборонний', result
            elif result <= 0.7:
                return 'Попереджувальний', result
            else:
                return 'Інформаційний', result
        except:
            return 'Невідомий', 0

def demonstrate_expert_system():
    print("1. ЕКСПЕРТНА СИСТЕМА (продукційні правила):")
    expert = TrafficSignExpertSystem()
    
    test_facts = {'color': 'red', 'shape': 'octagon', 'has_number': False}
    result = expert.infer(test_facts)
    print(f"   Вхід: колір=червоний, форма=восьмикутник")
    print(f"   Результат: {result['sign_type']} - {result['sign_name']}")

def demonstrate_semantic_network():
    print("\n2. СЕМАНТИЧНА МЕРЕЖА:")
    semantic_net = SemanticNetwork()
    relations = [
        ('stop_sign', 'is_a', 'prohibitory'),
        ('speed_limit', 'is_a', 'prohibitory'),
        ('yield_sign', 'is_a', 'warning'),
        ('stop_sign', 'has_color', 'red')
    ]
    
    for rel in relations:
        semantic_net.add_relation(*rel)

    results = semantic_net.query('stop_sign')
    print("   Запит зв'язків для 'stop_sign':")
    for rel in results:
        print(f"   {rel['from']} --{rel['relation']}--> {rel['to']}")

def demonstrate_frame_system():
    print("\n3. ФРЕЙМОВА СИСТЕМА:")
    traffic_sign = Frame("TrafficSign")
    traffic_sign.set_slot("призначення", "регулювання руху")
    prohibitory_sign = Frame("ProhibitorySign", traffic_sign)
    prohibitory_sign.set_slot("колір", "червоний")
    stop_sign = Frame("StopSign", prohibitory_sign)
    stop_sign.set_slot("форма", "восьмикутник")
    
    print("   Властивості стоп-знака:")
    print(f"   Призначення: {stop_sign.get_slot('призначення')}")
    print(f"   Колір: {stop_sign.get_slot('колір')}")
    print(f"   Форма: {stop_sign.get_slot('форма')}")

def demonstrate_neural_network():
    print("\n4. НЕЙРОННА МЕРЕЖА:")
    nn = TrafficSignNeuralNetwork()
    train_acc, test_acc = nn.train()
    print(f"   Точність навчання: {train_acc:.3f}")
    print(f"   Точність тесту: {test_acc:.3f}")

def demonstrate_fuzzy_system():
    print("\n5. НЕЧІТКА СИСТЕМА:")
    fuzzy_system = FuzzyTrafficSignSystem()
    
    test_cases = [
        (0.9, 0.8, 0.1),  # Червоний колір, кругла форма
        (0.8, 0.1, 0.9),  # Червоний колір, трикутна форма
        (0.2, 0.6, 0.2),  # Синій колір, кругла форма
        (0.5, 0.4, 0.5)   # Невизначений випадок
    ]
    
    for i, (color_red, shape_circle, shape_triangle) in enumerate(test_cases, 1):
        result, confidence = fuzzy_system.classify(color_red, shape_circle, shape_triangle)
        print(f"   Тест {i}: колір={color_red:.1f}, круг={shape_circle:.1f}, трикутник={shape_triangle:.1f}")
        print(f"   Результат: {result} (впевненість: {confidence:.3f})")

def comparative_analysis():
    print("\n6. ПОРІВНЯЛЬНИЙ АНАЛІЗ:")
    print("   Нечітка система vs Продукційні правила:")
    print("   - Нечітка система краще працює з неповними даними")
    print("   - Нечітка система надає ступінь впевненості")
    print("   - Продукційні правила більш зрозумілі та інтерпретовані")
    print("   - Нечітка система краще обробляє граничні випадки")

if __name__ == "__main__":
    print("=== СИСТЕМА РОЗПІЗНАВАННЯ ДОРОЖНІХ ЗНАКІВ ===")
    print("Демонстрація різних моделей представлення знань:\n")
    
    demonstrate_expert_system()
    demonstrate_semantic_network()
    demonstrate_frame_system()
    demonstrate_neural_network()
    demonstrate_fuzzy_system()
    comparative_analysis()
    
    print("\n=== ДЕМОНСТРАЦІЯ ЗАВЕРШЕНА ===")