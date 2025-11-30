import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random

class TrafficSignExpertSystem:
    def __init__(self):
        self.rules = [
            {"condition": "color_red > 0.7 and shape_circle > 0.7", "action": "sign_type = 'prohibitory'"},
            {"condition": "color_red > 0.7 and shape_triangle > 0.7", "action": "sign_type = 'warning'"},
            {"condition": "color_blue > 0.7 and shape_rectangle > 0.7", "action": "sign_type = 'information'"},
            {"condition": "shape_octagon > 0.7", "action": "sign_type = 'stop_sign'"},
            {"condition": "has_number > 0.7", "action": "sign_name = 'SPEED_LIMIT'"},
            {"condition": "color_red > 0.7 and shape_octagon > 0.7", "action": "sign_name = 'STOP'"}
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
        self.nodes = {
            'stop_sign': {'type': 'sign', 'meaning': 'Повна зупинка транспорту'},
            'speed_limit': {'type': 'sign', 'meaning': 'Обмеження максимальної швидкості'},
            'yield_sign': {'type': 'sign', 'meaning': 'Поступитися дорогою'},
            'prohibitory': {'type': 'category', 'description': 'Знаки заборон'},
            'warning': {'type': 'category', 'description': 'Попереджувальні знаки'},
            'information': {'type': 'category', 'description': 'Інформаційні знаки'}
        }
    
    def add_relation(self, from_node, relation_type, to_node):
        self.relations.append({'from': from_node, 'relation': relation_type, 'to': to_node})
    
    def query(self, start_node, relation_type=None):
        return [rel for rel in self.relations if rel['from'] == start_node and (relation_type is None or rel['relation'] == relation_type)]
    
    def get_node_info(self, node_id):
        return self.nodes.get(node_id, {})

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
        X = np.random.randn(n_samples, 7)
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
        class_names = ['prohibitory', 'warning', 'information']
        return {
            'class': class_names[int(prediction)],
            'probabilities': probabilities
        }

class IntegratedTrafficSignSystem:
    def __init__(self):
        self.expert_system = TrafficSignExpertSystem()
        self.semantic_net = SemanticNetwork()
        self.neural_net = TrafficSignNeuralNetwork()
        self.frame_system = self._initialize_frames()
        
        self._initialize_semantic_network()
        self.neural_net.train()
    
    def _initialize_frames(self):
        traffic_sign = Frame("TrafficSign")
        traffic_sign.set_slot("призначення", "регулювання дорожнього руху")
        traffic_sign.set_slot("розташування", "біля проїзної частини")
        
        prohibitory_sign = Frame("ProhibitorySign", traffic_sign)
        prohibitory_sign.set_slot("колір", "червоний")
        prohibitory_sign.set_slot("дія", "заборона")
        
        warning_sign = Frame("WarningSign", traffic_sign)
        warning_sign.set_slot("колір", "червоний")
        warning_sign.set_slot("форма", "трикутник")
        warning_sign.set_slot("дія", "попередження")
        
        information_sign = Frame("InformationSign", traffic_sign)
        information_sign.set_slot("колір", "синій")
        information_sign.set_slot("дія", "інформування")
        
        stop_sign = Frame("StopSign", prohibitory_sign)
        stop_sign.set_slot("форма", "восьмикутник")
        stop_sign.set_slot("вимога", "обов'язкова зупинка")
        
        speed_limit_sign = Frame("SpeedLimitSign", prohibitory_sign)
        speed_limit_sign.set_slot("форма", "круг")
        speed_limit_sign.set_slot("характеристика", "має числове значення")
        
        return {
            'traffic_sign': traffic_sign,
            'prohibitory': prohibitory_sign,
            'warning': warning_sign,
            'information': information_sign,
            'stop': stop_sign,
            'speed_limit': speed_limit_sign
        }
    
    def _initialize_semantic_network(self):
        relations = [
            ('stop_sign', 'is_a', 'prohibitory'),
            ('speed_limit', 'is_a', 'prohibitory'),
            ('yield_sign', 'is_a', 'warning'),
            ('prohibitory', 'category_of', 'traffic_signs'),
            ('warning', 'category_of', 'traffic_signs'),
            ('information', 'category_of', 'traffic_signs'),
            ('stop_sign', 'requires_action', 'full_stop'),
            ('speed_limit', 'regulates', 'vehicle_speed'),
            ('yield_sign', 'requires_action', 'give_way')
        ]
        
        for rel in relations:
            self.semantic_net.add_relation(*rel)
    
    def process_sign(self, features):
        results = {}
        
        results['neural_network'] = self.neural_net.predict(features)
        
        facts = {
            'color_red': features[0],
            'color_blue': features[1],
            'shape_circle': features[2],
            'shape_triangle': features[3],
            'shape_rectangle': features[4],
            'shape_octagon': features[5],
            'has_number': features[6],
            'sign_type': None,
            'sign_name': None
        }
        
        results['expert_system'] = self.expert_system.infer(facts)
        
        sign_type = results['expert_system'].get('sign_type', results['neural_network']['class'])
        results['semantic_network'] = self.semantic_net.query(sign_type)
        
        frame_key = 'stop' if sign_type == 'stop_sign' else sign_type
        frame = self.frame_system.get(frame_key, self.frame_system['traffic_sign'])
        results['frame_info'] = {
            'призначення': frame.get_slot('призначення'),
            'дія': frame.get_slot('дія'),
            'колір': frame.get_slot('колір'),
            'форма': frame.get_slot('форма')
        }
        
        return results
    
    def comprehensive_analysis(self, features):
        results = self.process_sign(features)
        
        analysis = {
            'predictive_power': f"Нейронна мережа: точність {results['neural_network']['probabilities'].max():.3f}",
            'explainability': f"Продукційні правила: {results['expert_system']}",
            'structured_knowledge': f"Фрейми: {results['frame_info']}",
            'semantic_relations': f"Семантична мережа: {len(results['semantic_network'])} зв'язків знайдено"
        }
        
        return analysis

def main():
    system = IntegratedTrafficSignSystem()
    
    test_cases = [
        [0.9, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1],
        [0.8, 0.1, 0.8, 0.1, 0.1, 0.1, 0.8],
        [0.1, 0.9, 0.1, 0.1, 0.8, 0.1, 0.2]
    ]
    
    for i, features in enumerate(test_cases, 1):
        print(f"Тестовий випадок {i}:")
        print(f"Вхідні дані: {features}")
        
        results = system.process_sign(features)
        analysis = system.comprehensive_analysis(features)
        
        print("Результати обробки:")
        print(f"  Нейронна мережа: {results['neural_network']['class']}")
        print(f"  Експертна система: {results['expert_system']}")
        print(f"  Фреймова інформація: {results['frame_info']}")
        
        print("Аналіз моделей:")
        for key, value in analysis.items():
            print(f"  {key}: {value}")
        print()

if __name__ == "__main__":
    main()