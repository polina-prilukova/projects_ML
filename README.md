## Проекты

1. **NLP**. [Анализ тональности отзывов](https://github.com/polina-prilukova/music_reviews_classifying)  
  **Задача**: разработать модель, которая будет классифицировать отзывы на товары в зависимости от эмоциональной окраски текстов этих отзывов.  
  **Предметная область**: продажи музыкальных инструментов и оборудования.  
  **Краткое описание**: Датасет сформирован вручную, данные собраны с сайтов магазинов музыкальных инструментов. 
  Для построения эмбеддингов текстов использован [universal sentence encoder](https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3).
  Рассмотрены различные подходы: мультиклассовая классификация с помощью алгоритмов sklearn, с использованием полносвязной нейросети на keras
  и путем дообучения нейросети [RuBert](https://huggingface.co/DeepPavlov/rubert-base-cased).  
  **Результат**: наилучший у модели на основе RuBert. ROC AUC score для каждого из классов в пределах 0.89-0.95.  
  **Основные библиотеки**: numpy, pandas, bs4, matplotlib, seaborn, nltk, sklearn, tensorflow, transformers
  
2. **Computer vision**. [Классификация изображений](https://github.com/polina-prilukova/projects_ML/blob/main/Images_classifying/dogs_cats_classification.ipynb)  
  **Задача**: разработать модель бинарной классификации изображений (кошки/собаки).  
  **Исходные данные**: датасет с [Kaggle](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition).  
  **Краткое описание**: на имеющемся наборе изображений обучались простая сверточная нейросеть, нейросеть на основе VGG-16 и нейросеть на основе ResNet-50.  
  **Результат**: Val accuracy = 0.9378, LogLoss (метрика оценки на Kaggle) = 0.18204  
  **Основные библиотеки**:  numpy, pandas, tensorflow, opencv

3. [**Recommender systems**](https://github.com/polina-prilukova/projects_ML/blob/main/Rec_system/Rec_system.ipynb)  
  **Задача**: разработать гибридную рекомендательную систему.  
  **Предметная область**: отзывы на кинофильмы. Датасет [ml-latest](https://grouplens.org/datasets/movielens/latest/).  
  **Краткое описание**: рекомендательная система выдает пользователю список фильмов на основе его предпочтений. Основные шаги при подборе фильмов:
 для просматриваемого пользователя составить топ его просмотров, подобрать похожие непросмотренные фильмы (алгоритм к-neibourghs), оценить эту подборку (задача регрессии),
выдать n фильмов с наиболее высокой оценкой.
  **Основные библиотеки**: numpy, pandas, sklearn, surprise
  
  4. **Classical ML**  
  * [Регрессия](https://github.com/polina-prilukova/projects_ML/blob/main/Regression/Regression.ipynb)    
    **Исходные данные**: данные о поездках службы такси Uber. Датасет с [Kaggle](https://www.kaggle.com/yasserh/uber-fares-dataset)    
    **Задача**: построить модель регрессии для предсказания цены.  
    **Краткое описание**: данные обработаны от выбросов, добавлены новые признаки. Использовались модели из пакетов sklearn и xgboost: простые и ансамбли моделей.   
    Для наиболее результативных моделей осуществлен подбор гиперпараметров.  
    **Результат**: R2-score = 0.69  
    **Основные библиотеки**: numpy, pandas, matplotlib, seaborn, sklearn, xgboost  

  * [Кластеризация](https://nbviewer.org/github/polina-prilukova/projects_ML/blob/main/Clustering/Geo_comment_clustering.ipynb)    
    **Исходные данные**: геокоординаты с публичных слушаний Москвы по правилам землепользования и застройки (ПЗЗ).    
    **Задача**: визуально разделить город на районы безотносительно голосов    
    **Краткое описание**: рассмотрены различные алгоритмы кластеризации (K-means, DBSCAN, Birch и др.). Для K-means рассчитано оптимальное количество кластеров, 
    для прочих алгоритмов параметры подобраны с учетом наилучшего полученного Silhouette score. Кластеризованные геометки нанесены на карту Москвы.  
    **Основные библиотеки**: numpy, pandas, matplotlib, sklearn, folium  

  * [EDA](https://github.com/polina-prilukova/projects_ML/blob/main/EDA/911_calls_EDA.ipynb)  
    **Задача**: Провести базовый EDA выбранного набора данных   
    **Исходные данные**: данные о звонках в экстренные службы города Монтгомери, Пенсильвания. Датасет с [Kaggle](https://www.kaggle.com/mchirico/montcoalert).    
    **Краткое описание**: данные обработаны от выбросов и пропусков, добавлены новые признаки, построены визуализации.   
    **Основные библиотеки**: numpy, pandas, matplotlib, sklearn
