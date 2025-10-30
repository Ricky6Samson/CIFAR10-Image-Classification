use CIFAR10

create table Model_Info(
model_id char(25),
model_name char(25),
val_accuracy float,
train_accuracy float,
date_trained date,
note varchar(100)
)

create table Epoch_Stats(
model_id char(25),
epoch int,
train_accuracy float,
val_accuracy float,
train_loss float,
val_loss float
)

create table Class_Metrics(
model_id char(25),
class_name char(25),
[precision] float,
recall float,
f1_score float,
support int
)

create table Confusion_Matrix(
model_id char(25),
true_class char(10),
predicted_class char(10),
[count] int
)

CREATE TABLE Predictions_Log (
    prediction_id int identity(1,1),
    model_id char(25),
    image_id int,
    true_class char(10),
    predicted_class char(10),
    true_class_index int,
    predicted_class_index int,
    confidence_score float,
    prediction_latency_ms float,
    correct bit
)

select * from Predictions_Log

