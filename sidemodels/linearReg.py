import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,accuracy_score
#import tensorflow as tf

df=pd.read_csv("data/insurance.csv")
print(df.head)
num_cols=df.select_dtypes("number").columns
cat_cols=df.select_dtypes(exclude="number").columns
scaler=StandardScaler()
label_encoder=LabelEncoder()
df[num_cols]=df[num_cols].fillna(df[num_cols].median)
df[cat_cols]=df[cat_cols].fillna("unknown")
df[num_cols]=scaler.fit_transform(df[num_cols])
for cat in cat_cols:
    df[cat]=label_encoder.fit_transform(df[cat])

df=df[num_cols].fillna(df[num_cols].median)

X=df.drop(columns="charges")
y=df['charges']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42 )
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVM": SVR()
}
print(models.items())

from sklearn.metrics import mean_squared_error, r2_score

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model: {name} | MSE: {mse:.4f} | R²: {r2:.4f}")


#Using tensorflow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1)  # Linear activation by default
])

model.compile(
    loss="mse",
    metrics=["mae"],
    optimizer="adam"
)

history=model.fit(
    X_train,y_train,
    validation_split=0.2,
    epochs=30,
    verbose=1,
    batch_size=32
)
loss,mae=model.evaluate(X_test,y_test)
y_pred = model.predict(X_test)


import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

