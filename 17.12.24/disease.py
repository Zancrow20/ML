import pandas as pd
import numpy as np

def main():
    # Чтение CSV файла
    symptoms = pd.read_csv('symptom.csv', sep=';')
    diseases = pd.read_csv('disease.csv', sep=';')

    # симптомы
    complains = []

    # рандомно выдаем симптомы
    for i in range(symptoms.shape[0]):
        complains.append(np.random.randint(0,2))


    disease_probability = np.ones(diseases.shape[0]-1)

    for i in range(len(disease_probability)):
        # вычисляем полную вероятность болезни по таблице с болезнями пациентов (с количеством)
        disease_probability[i] *= diseases.loc[i].values[1] / diseases.loc[diseases.shape[0]-1].values[1]
        for j in range(len(complains)):
            # проходимся по каждому симптому и вычисляем вероятность болезни.
            # Если complains[j] == 1, то умножаем на условную вероятность
            # Если complains[j] == 0, то умножаем на 1 - условную вероятность
            disease_probability[i] *= symptoms.loc[j].values[i+1] if complains[j] == 1 else 1 # 1 - symptoms.loc[j].values[i+1]
    print(diseases.loc[np.argmax(disease_probability)].values[0])

if __name__ == '__main__':
    main()