# from django.contrib.postgres.fields import JSONField
from django.db import models

# Create your models here.
import datetime
# Create your models here.
from django.contrib.auth.models import AbstractUser
from django.db.models import JSONField


class User(AbstractUser):
    class Meta:
        # Add this meta option to prevent clash with auth.User's related_name
        app_label = 'snowAIWeb'

    # Add related_names to fields causing the conflict
    groups = None  # To remove the reverse accessor
    user_permissions = None 


class TellUsMore(models.Model):
    user_email = models.EmailField()
    trading_experience = models.CharField(max_length=150)
    main_assets = models.CharField(max_length=2000)
    initial_capital = models.FloatField()
    trading_goals = models.CharField(max_length=1000)
    benefits = models.CharField(max_length=1000)


class Trade(models.Model):
    email = models.EmailField()
    asset = models.CharField(max_length=100)
    order_type = models.CharField(max_length=50)
    strategy = models.CharField(max_length=500)
    lot_size = models.FloatField()
    timeframe = models.CharField(max_length=100)
    roi = models.FloatField()
    entry_date = models.DateTimeField()
    exit_date = models.DateTimeField()
    entry_point = models.FloatField()
    stop_loss = models.FloatField()
    take_profit = models.FloatField()
    exit_point = models.FloatField()
    outcome = models.CharField(max_length=200)
    amount = models.FloatField()
    emotional_bias = models.CharField(max_length=100)
    reflection = models.CharField(max_length=10000)


class Journal(models.Model):
    user_email = models.EmailField()
    content = models.CharField(max_length=10000)
    created_date = models.DateTimeField()


class Journals(models.Model):
    user_email = models.EmailField()
    content = models.CharField(max_length=10000)
    created_date = models.DateTimeField()
    tags = models.CharField(max_length=300, default="")

    
class NewsData(models.Model):
    data = JSONField()  # Store the entire row data as a JSON object
    created_at = models.DateTimeField(auto_now_add=True)


class Conversations(models.Model):
    user_email = models.EmailField()
    conversation_id = models.CharField(max_length=100)
    conversation = models.TextField()  # Use TextField to store longer text data


class Conversation(models.Model):
    user_email = models.EmailField()
    conversation_id = models.CharField(max_length=100)
    conversation = models.TextField()  # Use TextField to store longer text data


class News(models.Model):
    user_email = models.EmailField()
    symbol = models.CharField(max_length=100)
    data = models.TextField()
    day_created = models.DateTimeField()


class CreateModel(models.Model):
    bot_type = models.CharField(max_length=200)
    params = models.CharField(max_length=200)
    

class MovingAverageBot(models.Model):
    # user_email = models.EmailField()
    ma1_type = models.CharField(max_length=20)
    ma2_type = models.CharField(max_length=20)
    ma1 = models.IntegerField()
    ma2 = models.IntegerField()


class Bot(models.Model):
    username = models.EmailField()
    magic_number = models.IntegerField()
    time_saved = models.DateTimeField()
    parameters = models.TextField()


class SaveDataset(models.Model):
    dataset = models.CharField(max_length=30)


class ContactUs(models.Model):
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    email = models.EmailField()
    message = models.TextField()