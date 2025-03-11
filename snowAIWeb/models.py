# from django.contrib.postgres.fields import JSONField
from django.db import models
import datetime
from django.contrib.auth.models import AbstractUser
from django.db.models import JSONField
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager


class CustomUserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save()
        return user


class CustomUser(AbstractBaseUser):
    email = models.EmailField(unique=True)
    username = models.CharField(max_length=50, unique=True)
    # Add other fields (e.g., first_name, last_name) if needed

    objects = CustomUserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']
    

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


class Account(models.Model):
    account_name = models.CharField(max_length=100, unique=True)  # Unique account identifier
    main_assets = models.CharField(max_length=255)  # Main asset classes, e.g., Forex, Equities
    initial_capital = models.FloatField()  # Initial capital as a float

    def __str__(self):
        return self.account_name


class AccountTrades(models.Model):
    account = models.ForeignKey(
        Account, related_name='trades', on_delete=models.CASCADE
    )  # Link to Account
    asset = models.CharField(max_length=100)  # Traded asset, e.g., EURUSD, XAUUSD
    order_type = models.CharField(max_length=50)  # Type of order, e.g., Buy or Sell
    strategy = models.CharField(max_length=100)  # Strategy used for the trade
    day_of_week_entered = models.CharField(max_length=10)  # Day trade was entered, e.g., Monday
    day_of_week_closed = models.CharField(max_length=10, blank=True, null=True)  # Day trade closed
    trading_session_entered = models.CharField(max_length=50)  # Session entered, e.g., London, NY
    trading_session_closed = models.CharField(max_length=50, blank=True, null=True)  # Session closed
    outcome = models.CharField(max_length=10)  # Profit or Loss
    amount = models.FloatField()  # Trade amount as a float, e.g., -100 or 150
    emotional_bias = models.TextField(blank=True, null=True)  # Notes on emotional state (optional)
    reflection = models.TextField(blank=True, null=True)  # Reflective notes (optional)
    date_entered = models.DateTimeField(blank=True, null=True)  # New field for the date trade was entered

    def __str__(self):
        return f"{self.account.account_name} - {self.asset} ({self.order_type})"


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


class SplitDataset(models.Model):
    start_year = models.CharField(max_length=30)
    end_year = models.CharField(max_length=30)


class SetInitCapital(models.Model):
    initial_capital = models.FloatField()


class GenesysLive(models.Model):
    model_id = models.IntegerField(unique=True)
    model_code = models.TextField()
    true_initial_equity = models.FloatField()
    # current_equity = models.FloatField()
    # take_profit_number = models.FloatField()
    # take_profit_type = models.CharField(max_length=20)
    # stop_loss_number = models.FloatField()
    # stop_loss_type = models.CharField(max_length=20)


class tradeModel(models.Model):
    model_id = models.IntegerField()
    model_code = models.TextField()
    initial_equity = models.FloatField()
    order_ticket = models.TextField()
    asset = models.CharField(max_length=20, null=True)
    profit = models.FloatField(null=True)
    volume = models.FloatField(null=True)
    type_of_trade = models.CharField(max_length=10, null=True)
    timeframe = models.CharField(max_length=10, null=True)
    date_taken = models.DateTimeField(null=True)


class uniqueBot(models.Model):
    model_id = models.IntegerField()
    order_ticket = models.TextField()
    asset = models.CharField(max_length=20, null=True)
    bot_id = models.TextField()


class dailyBrief(models.Model):
    asset = models.TextField()
    summary = models.TextField()
    last_update = models.DateTimeField(null=True)


class DailyBriefAssets(models.Model):
    asset = models.CharField(max_length=50)  # Use CharField for asset names or codes


class Chill(models.Model):
    section = models.TextField()
    text = models.TextField()


class AlertBot(models.Model):
    CONDITION_CHOICES = [
        ('<', 'Less than'),
        ('>', 'Greater than'),
        ('=', 'Equal to'),
    ]

    asset = models.CharField(max_length=50)  # Use CharField for asset names or codes
    price = models.FloatField()  # FloatField for prices
    condition = models.CharField(max_length=1, choices=CONDITION_CHOICES)  # Limit choices to valid conditions
    checked = models.BooleanField(default=False)  # BooleanField for simple checked/unchecked state

    def __str__(self):
        return f"{self.asset} {self.condition} {self.price}"

    class Meta:
        verbose_name = "Alert"
        verbose_name_plural = "Alerts"


class BacktestModels(models.Model):
    chosen_dataset = models.CharField(max_length=255, help_text="Name or path of the dataset used.")
    generated_code = models.TextField(help_text="Generated code for the backtest.")
    model_backtested = models.BooleanField(default=False, help_text="Indicates whether the model was backtested.")
    dataset_start = models.CharField(max_length=50, help_text="Start date of the dataset (YYYY-MM-DD format).")
    dataset_end = models.CharField(max_length=50, help_text="End date of the dataset (YYYY-MM-DD format).")
    initial_capital = models.FloatField(help_text="Initial capital for the backtest.")

    def __str__(self):
        return f"Backtest on {self.chosen_dataset} from {self.dataset_start} to {self.dataset_end}"


class BacktestResult(models.Model):

    backtest_model = models.ForeignKey(BacktestModels, on_delete=models.CASCADE, help_text="Related backtest model", null=True)

    start = models.DateField(help_text="Start date of the backtest.")
    end = models.DateField(help_text="End date of the backtest.")
    duration = models.CharField(max_length=100, help_text="Duration of the backtest.")

    exposure_time = models.FloatField(help_text="Exposure time percentage.")
    equity_final = models.FloatField(help_text="Final equity in dollars.")
    equity_peak = models.FloatField(help_text="Peak equity in dollars.")

    return_percent = models.FloatField(help_text="Total return percentage.")
    buy_hold_return = models.FloatField(help_text="Buy & Hold return percentage.")
    annual_return = models.FloatField(help_text="Annualized return percentage.")

    volatility_annual = models.FloatField(help_text="Annualized volatility percentage.")
    sharpe_ratio = models.FloatField(help_text="Sharpe ratio of the strategy.")
    sortino_ratio = models.FloatField(help_text="Sortino ratio of the strategy.")
    calmar_ratio = models.FloatField(help_text="Calmar ratio of the strategy.")

    max_drawdown = models.FloatField(help_text="Maximum drawdown percentage.")
    avg_drawdown = models.FloatField(help_text="Average drawdown percentage.")
    max_drawdown_duration = models.CharField(max_length=100, help_text="Maximum drawdown duration.")
    avg_drawdown_duration = models.CharField(max_length=100, help_text="Average drawdown duration.")

    num_trades = models.IntegerField(help_text="Total number of trades.")
    win_rate = models.FloatField(help_text="Win rate percentage.")

    best_trade = models.FloatField(help_text="Best trade percentage.")
    worst_trade = models.FloatField(help_text="Worst trade percentage.")
    avg_trade = models.FloatField(help_text="Average trade percentage.")

    max_trade_duration = models.CharField(max_length=100, help_text="Maximum trade duration.")
    avg_trade_duration = models.CharField(max_length=100, help_text="Average trade duration.")

    profit_factor = models.FloatField(help_text="Profit factor of the strategy.")
    expectancy = models.FloatField(help_text="Expectancy percentage.")

    plot_json = models.JSONField(blank=True, null=True, help_text="Plot data in JSON format.")

    created_at = models.DateTimeField(auto_now_add=True, help_text="Record creation timestamp.")

    def __str__(self):
        return f"Backtest from {self.start} to {self.end}"


class IdeaModel(models.Model):
    idea_category = models.CharField(max_length=255)  # A text field for the category
    idea_text = models.TextField()  # A larger text field for the idea description
    idea_tracker = models.CharField(
        max_length=50, 
        choices=[
            ('Pending', 'Pending'), 
            ('In Progress', 'In Progress'), 
            ('Completed', 'Completed')
        ]
    )  # A dropdown for status tracking
    created_at = models.DateTimeField(auto_now_add=True)  # Automatically set to the current time on creation

    def __str__(self):
        return self.idea_text[:50]  # Show the first 50 characters of the idea text


class SavedQuiz(models.Model):
    # user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    quiz_name = models.CharField(max_length=255)
    total_questions = models.IntegerField()
    correct_answers = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.quiz_name} - {self.correct_answers}/{self.total_questions}"


class SavedQuizQuestion(models.Model):
    saved_quiz = models.ForeignKey(SavedQuiz, related_name='questions', on_delete=models.CASCADE)
    question = models.TextField()
    selected_answer = models.CharField(max_length=255)
    correct_answer = models.CharField(max_length=255)
    is_correct = models.BooleanField()

    def __str__(self):
        return f"Question for {self.saved_quiz.quiz_name}"


class MusicModel(models.Model):
    name = models.CharField(max_length=255)
    file = models.FileField()  # No upload_to specified, it will use the default media directory
    
    def __str__(self):
        return self.name


class FeedbackForm(models.Model): 
    feedback = models.TextField()







class ContactUs(models.Model):
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    email = models.EmailField()
    message = models.TextField()


class BookOrder(models.Model):
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    interested_product = models.TextField()
    email = models.EmailField()
    phone_number = models.CharField(max_length=20)
    number_of_units = models.IntegerField()