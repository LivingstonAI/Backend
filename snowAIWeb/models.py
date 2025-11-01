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
    benefits = models.CharField(max_length=1001)


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
    file_data = models.BinaryField(null=True, blank=True)  # Allow null values
    content_type = models.CharField(max_length=100, default='audio/mpeg')
    file_name = models.CharField(max_length=255, null=True)  # Store original filename
    created_at = models.DateTimeField(auto_now_add=True, null=True)
    updated_at = models.DateTimeField(auto_now=True, null=True)

    def __str__(self):
        return self.name

from django.utils import timezone

class TradeIdea(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('executed', 'Executed'),
        ('closed', 'Closed'),
        ('cancelled', 'Cancelled'),
    ]
    
    OUTCOME_CHOICES = [
        ('pending', 'Pending'),
        ('win', 'Win'),
        ('loss', 'Loss'),
        ('breakeven', 'Breakeven'),
    ]
    
    heading = models.CharField(max_length=200)
    asset = models.CharField(max_length=100)
    trade_idea = models.TextField()
    date_created = models.DateTimeField(default=timezone.now)
    trade_status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    target_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    stop_loss = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    entry_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    outcome = models.CharField(max_length=20, choices=OUTCOME_CHOICES, default='pending')
    
    def __str__(self):
        return f"{self.heading} - {self.asset}"
        

class AssetsTracker(models.Model):
    asset = models.CharField(max_length=50)


class PropFirm(models.Model):
    name = models.CharField(max_length=100)
    logo = models.TextField()  # For base64 encoded image
    website = models.URLField(blank=True, null=True)
    
    def __str__(self):
        return self.name

class PropFirmAccount(models.Model):
    ACCOUNT_TYPES = (
        ('CHALLENGE', 'Challenge'),
        ('VERIFICATION', 'Verification'),
        ('FUNDED', 'Funded'),
    )
    STATUS_CHOICES = (
        ('IN_PROGRESS', 'In Progress'),
        ('PASSED', 'Passed'),
        ('FAILED', 'Failed'),
        ('LIVE', 'Live'),
    )
    prop_firm = models.ForeignKey(PropFirm, on_delete=models.CASCADE, related_name='firm_accounts')
    account_name = models.CharField(max_length=100)
    account_id = models.CharField(max_length=100, blank=True, null=True)
    account_type = models.CharField(max_length=20, choices=ACCOUNT_TYPES)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='IN_PROGRESS')
    
    # Financial metrics
    initial_balance = models.DecimalField(max_digits=15, decimal_places=2)
    current_balance = models.DecimalField(max_digits=15, decimal_places=2)
    current_equity = models.DecimalField(max_digits=15, decimal_places=2)
    
    # Risk parameters
    daily_loss_limit = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True)
    max_loss_limit = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True)
    profit_target = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True)
    
    # Time constraints
    start_date = models.DateField()
    end_date = models.DateField(null=True, blank=True)  # For challenges with time limits
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.prop_firm.name} - {self.account_name}"
    
    def days_remaining(self):
        if not self.end_date:
            return None
        from datetime import date
        delta = self.end_date - date.today()
        return max(0, delta.days)
    
    def percentage_to_target(self):
        if not self.profit_target or self.profit_target == 0:
            return None
        profit = self.current_balance - self.initial_balance
        return (profit / self.profit_target) * 100

class TradingDay(models.Model):
    account = models.ForeignKey(PropFirmAccount, on_delete=models.CASCADE, related_name='trading_days')
    date = models.DateField()
    starting_balance = models.DecimalField(max_digits=15, decimal_places=2)
    ending_balance = models.DecimalField(max_digits=15, decimal_places=2)
    pnl = models.DecimalField(max_digits=15, decimal_places=2)
    session_time_minutes = models.IntegerField(default=0)
    notes = models.TextField(blank=True, null=True)
    voice_memo = models.FileField(upload_to='voice_memos/', null=True, blank=True)
    
    class Meta:
        unique_together = ('account', 'date')
    
    def __str__(self):
        return f"{self.account} - {self.date} (${self.pnl})"

class PropTrade(models.Model):
    TRADE_TYPES = (
        ('BUY', 'Buy'),
        ('SELL', 'Sell'),
    )
    
    account = models.ForeignKey(PropFirmAccount, on_delete=models.CASCADE, related_name='trades')
    trading_day = models.ForeignKey(TradingDay, on_delete=models.CASCADE, related_name='trades', null=True, blank=True)
    asset = models.CharField(max_length=50)
    trade_type = models.CharField(max_length=4, choices=TRADE_TYPES)
    entry_price = models.DecimalField(max_digits=15, decimal_places=5)
    exit_price = models.DecimalField(max_digits=15, decimal_places=5, null=True, blank=True)
    size = models.DecimalField(max_digits=15, decimal_places=5)
    entry_time = models.DateTimeField()
    exit_time = models.DateTimeField(null=True, blank=True)
    pnl = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True)
    strategy = models.CharField(max_length=100, blank=True, null=True)
    notes = models.TextField(blank=True, null=True)
    
    def __str__(self):
        return f"{self.asset} {self.trade_type} - {self.entry_time}"

class ManagementMetrics(models.Model):
    total_accounts = models.IntegerField(default=0)
    total_capital_managed = models.DecimalField(max_digits=15, decimal_places=2, default=0)
    total_profit = models.DecimalField(max_digits=15, decimal_places=2, default=0)
    win_rate = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    avg_risk_reward = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    avg_session_time = models.IntegerField(default=0)  # in minutes
    last_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return "Prop Firm Management Metrics"

# models.py
from django.db import models


class PropFirmManagementMetrics(models.Model):
    ACCOUNT_TYPE_CHOICES = [
        ('challenge', 'Challenge'),
        ('verification', 'Verification'),
        ('funded', 'Funded'),
    ]
    
    STATUS_CHOICES = [
        ('in_progress', 'In Progress'),
        ('passed', 'Passed'),
        ('failed', 'Failed'),
        ('live', 'Live'),
    ]
    
    prop_firm = models.ForeignKey(PropFirm, on_delete=models.CASCADE, related_name='accounts')
    account_type = models.CharField(max_length=20, choices=ACCOUNT_TYPE_CHOICES)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES)
    account_id = models.CharField(max_length=100, blank=True, null=True)
    starting_balance = models.DecimalField(max_digits=15, decimal_places=2)
    current_balance = models.DecimalField(max_digits=15, decimal_places=2)
    current_equity = models.DecimalField(max_digits=15, decimal_places=2)
    profit_target = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    max_drawdown = models.DecimalField(max_digits=15, decimal_places=2, blank=True, null=True)
    start_date = models.DateField()
    notes = models.TextField(blank=True, null=True)
    
    def __str__(self):
        return f"{self.prop_firm.name} - {self.account_type} - {self.status}"




class EconomicEvent(models.Model):
    IMPACT_CHOICES = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
    ]
    
    CURRENCY_CHOICES = [
        ('USD', 'US Dollar'),
        ('EUR', 'Euro'),
        ('GBP', 'British Pound'),
        ('JPY', 'Japanese Yen'),
        ('AUD', 'Australian Dollar'),
        ('CAD', 'Canadian Dollar'),
        ('CHF', 'Swiss Franc'),
        ('CNY', 'Chinese Yuan'),
    ]
    
    date_time = models.DateTimeField()
    currency = models.CharField(max_length=3, choices=CURRENCY_CHOICES)
    impact = models.CharField(max_length=10, choices=IMPACT_CHOICES)
    event_name = models.CharField(max_length=255)
    actual = models.CharField(max_length=50, blank=True, null=True)
    forecast = models.CharField(max_length=50, blank=True, null=True)
    previous = models.CharField(max_length=50, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['date_time']
    
    def __str__(self):
        return f"{self.date_time.strftime('%Y-%m-%d %H:%M')} - {self.currency} - {self.event_name}"

import uuid

class PaperGPT(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=500)
    file_name = models.CharField(max_length=255)
    file_data = models.TextField()  # Base64 encoded PDF
    file_size = models.BigIntegerField()
    extracted_text = models.TextField()
    ai_summary = models.TextField()
    category = models.CharField(max_length=100, blank=True, null=True)
    personal_notes = models.TextField(blank=True)
    upload_date = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    
    class Meta:
        ordering = ['-upload_date']



# Add these to your existing models.py
class TraderGPTForexAnalysisSession(models.Model):
    session_id = models.CharField(max_length=100, unique=True)
    user_email = models.EmailField()
    currency_pairs = models.JSONField()  # Store array of currency pairs
    analysis_timestamp = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, default='pending')  # pending, completed, failed
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Analysis Session {self.session_id} - {self.user_email}"


class TraderGPTForexAnalysisResult(models.Model):
    SENTIMENT_CHOICES = [
        ('bullish', 'Bullish'),
        ('bearish', 'Bearish'),
        ('neutral', 'Neutral'),
    ]
    
    RISK_CHOICES = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
    ]
    
    analysis_session = models.ForeignKey(TraderGPTForexAnalysisSession, on_delete=models.CASCADE, related_name='results')
    currency_pair = models.CharField(max_length=10)
    sentiment = models.CharField(max_length=10, choices=SENTIMENT_CHOICES)
    confidence_score = models.IntegerField()  # 0-100
    entry_strategy = models.TextField()
    risk_level = models.CharField(max_length=10, choices=RISK_CHOICES)
    time_horizon = models.CharField(max_length=50)
    target_price = models.CharField(max_length=20)
    stop_loss = models.CharField(max_length=20, blank=True, null=True)
    key_factors = models.TextField()
    technical_analysis = models.TextField(blank=True, null=True)
    fundamental_analysis = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.currency_pair} - {self.sentiment} ({self.confidence_score}%)"


class TraderGPTAnalysisNewsLink(models.Model):
    analysis_result = models.ForeignKey(TraderGPTForexAnalysisResult, on_delete=models.CASCADE, related_name='linked_news')
    title = models.CharField(max_length=500)
    description = models.TextField()
    source = models.CharField(max_length=100)
    url = models.URLField()
    highlights = models.TextField(blank=True, null=True)
    relevance_score = models.IntegerField(default=0)  # 0-100
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.analysis_result.currency_pair} - {self.title[:50]}..."


class TraderGPTAnalysisEconomicEventLink(models.Model):
    analysis_result = models.ForeignKey(TraderGPTForexAnalysisResult, on_delete=models.CASCADE, related_name='linked_economic_events')
    economic_event = models.ForeignKey(EconomicEvent, on_delete=models.CASCADE)
    relevance_score = models.IntegerField(default=0)  # 0-100
    impact_assessment = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.analysis_result.currency_pair} - {self.economic_event.event_name}"




# models.py - Add these to your existing models

from django.db import models
from django.utils import timezone

class WatchedTradingAsset(models.Model):
    """Model to store currency pairs that users want to analyze"""
    ASSET_CHOICES = [
        ('EURUSD', 'EUR/USD'),
        ('GBPUSD', 'GBP/USD'),
        ('USDJPY', 'USD/JPY'),
        ('USDCHF', 'USD/CHF'),
        ('AUDUSD', 'AUD/USD'),
        ('USDCAD', 'USD/CAD'),
        ('NZDUSD', 'NZD/USD'),
        ('EURJPY', 'EUR/JPY'),
        ('GBPJPY', 'GBP/JPY'),
        ('EURGBP', 'EUR/GBP'),
        ('AUDJPY', 'AUD/JPY'),
        ('EURAUD', 'EUR/AUD'),
        ('USDCNH', 'USD/CNH'),
        ('GBPAUD', 'GBP/AUD'),
        ('EURCHF', 'EUR/CHF'),
        ('AUDCAD', 'AUD/CAD'),
        ('GBPCAD', 'GBP/CAD'),
        ('EURCAD', 'EUR/CAD'),
    ]
    
    asset = models.CharField(max_length=10, choices=ASSET_CHOICES, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        ordering = ['asset']
    
    def __str__(self):
        return self.asset


class TraderGPTAnalysisRecord(models.Model):
    """Model to store TraderGPT analysis results"""
    SENTIMENT_CHOICES = [
        ('bullish', 'Bullish'),
        ('bearish', 'Bearish'),
        ('neutral', 'Neutral'),
    ]
    
    RISK_CHOICES = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
    ]
    
    TIME_HORIZON_CHOICES = [
        ('short', 'Short Term (1-7 days)'),
        ('medium', 'Medium Term (1-4 weeks)'),
        ('long', 'Long Term (1-6 months)'),
    ]
    
    asset = models.CharField(max_length=10)
    market_sentiment = models.CharField(max_length=10, choices=SENTIMENT_CHOICES)
    confidence_score = models.IntegerField(help_text="Confidence score from 1-100")
    risk_level = models.CharField(max_length=10, choices=RISK_CHOICES)
    time_horizon = models.CharField(max_length=10, choices=TIME_HORIZON_CHOICES)
    entry_strategy = models.TextField()
    key_factors = models.TextField()
    # Increased max_length for these fields to accommodate longer GPT responses
    stop_loss_level = models.CharField(max_length=200, blank=True, null=True)
    take_profit_level = models.CharField(max_length=200, blank=True, null=True)
    support_level = models.CharField(max_length=200, blank=True, null=True)
    resistance_level = models.CharField(max_length=200, blank=True, null=True)
    raw_analysis = models.TextField(help_text="Full GPT response")
    news_data_used = models.JSONField(default=dict, blank=True)
    economic_events_used = models.JSONField(default=dict, blank=True)
    analysis_timestamp = models.DateTimeField(default=timezone.now)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-analysis_timestamp']
        unique_together = ['asset', 'analysis_timestamp']
    
    def __str__(self):
        return f"{self.asset} - {self.market_sentiment} ({self.confidence_score}%) - {self.analysis_timestamp.strftime('%Y-%m-%d %H:%M')}"

        
class AnalysisExecutionLog(models.Model):
    """Model to log analysis execution attempts"""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    asset = models.CharField(max_length=10)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='pending')
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True, null=True)
    execution_time_seconds = models.FloatField(null=True, blank=True)
    
    class Meta:
        ordering = ['-started_at']
    
    def __str__(self):
        return f"{self.asset} - {self.status} - {self.started_at.strftime('%Y-%m-%d %H:%M')}"

# Add this to your models.py

import json
from django.db import models
from django.utils import timezone

class AITradingCouncilConversation(models.Model):
    """Model to store AI Trading Council conversations"""
    
    CONVERSATION_STATUS_CHOICES = [
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    ECONOMIC_OUTLOOK_CHOICES = [
        ('very_positive', 'Very Positive'),
        ('positive', 'Positive'),
        ('neutral', 'Neutral'),
        ('negative', 'Negative'),
        ('very_negative', 'Very Negative'),
    ]
    
    MARKET_VOLATILITY_CHOICES = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('extreme', 'Extreme'),
    ]
    
    # Basic conversation info
    conversation_id = models.CharField(max_length=100, unique=True)
    title = models.CharField(max_length=200, default="AI Trading Council Discussion")
    status = models.CharField(max_length=20, choices=CONVERSATION_STATUS_CHOICES, default='running')
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    # Participants and assets
    participating_assets = models.JSONField(default=list)  # List of assets that participated
    total_participants = models.IntegerField(default=0)
    
    # Conversation content
    conversation_data = models.JSONField(default=dict)  # Full conversation with turns
    conversation_summary = models.TextField(blank=True)
    
    # Economic insights and metrics
    overall_economic_outlook = models.CharField(
        max_length=20, 
        choices=ECONOMIC_OUTLOOK_CHOICES, 
        default='neutral'
    )
    global_market_sentiment = models.CharField(max_length=20, default='neutral')
    market_volatility_level = models.CharField(
        max_length=20, 
        choices=MARKET_VOLATILITY_CHOICES, 
        default='medium'
    )
    
    # Key insights
    major_economic_themes = models.JSONField(default=list)  # List of key themes discussed
    currency_strength_rankings = models.JSONField(default=dict)  # Currency strength analysis
    risk_factors_identified = models.JSONField(default=list)  # Risk factors mentioned
    opportunity_areas = models.JSONField(default=list)  # Opportunities identified
    
    # Metrics for quick analysis
    bullish_sentiment_count = models.IntegerField(default=0)
    bearish_sentiment_count = models.IntegerField(default=0)
    neutral_sentiment_count = models.IntegerField(default=0)
    average_confidence_score = models.FloatField(default=0.0)
    
    # Execution details
    execution_time_seconds = models.FloatField(default=0.0)
    error_message = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-created_at']
        db_table = 'ai_trading_council_conversations'
    
    def __str__(self):
        return f"Council Discussion {self.conversation_id} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
    
    def get_participant_count(self):
        """Get the number of participants in this conversation"""
        return len(self.participating_assets)
    
    def get_dominant_sentiment(self):
        """Get the most common sentiment from the conversation"""
        sentiments = {
            'bullish': self.bullish_sentiment_count,
            'bearish': self.bearish_sentiment_count,
            'neutral': self.neutral_sentiment_count
        }
        return max(sentiments, key=sentiments.get)
    
    def get_conversation_turns_count(self):
        """Get total number of turns in the conversation"""
        if isinstance(self.conversation_data, dict) and 'turns' in self.conversation_data:
            return len(self.conversation_data['turns'])
        return 0


class AITradingCouncilParticipant(models.Model):
    """Model to track individual participants in council conversations"""
    
    conversation = models.ForeignKey(
        AITradingCouncilConversation, 
        on_delete=models.CASCADE,
        related_name='participants'
    )
    asset_code = models.CharField(max_length=10)  # e.g., 'EURUSD'
    participant_name = models.CharField(max_length=100)  # e.g., 'EUR/USD Analyst'
    
    # Analysis data this participant contributed
    market_sentiment = models.CharField(max_length=20, default='neutral')
    confidence_score = models.IntegerField(default=50)
    risk_assessment = models.CharField(max_length=20, default='medium')
    
    # Key points made by this participant
    key_insights = models.JSONField(default=list)
    turns_spoken = models.IntegerField(default=0)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['conversation', 'asset_code']
        db_table = 'ai_trading_council_participants'
    
    def __str__(self):
        return f"{self.participant_name} in {self.conversation.conversation_id}"

from django.db import models
from django.utils import timezone
import uuid
import base64
from django.core.files.storage import default_storage

class FirmCompliance(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    firm_name = models.CharField(max_length=200, verbose_name="Firm Name")
    firm_logo = models.ImageField(upload_to='firm_logos/', null=True, blank=True, verbose_name="Firm Logo")
    personal_notes = models.TextField(verbose_name="Personal Notes", help_text="Your personal compliance notes and rules")
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "Firm Compliance"
        verbose_name_plural = "Firm Compliance Records"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.firm_name} - Compliance Rules"
    
    @property
    def logo_url(self):
        """Return the logo as base64 data URL if exists, otherwise return None"""
        if self.firm_logo and hasattr(self.firm_logo, 'url'):
            try:
                # Read the file and convert to base64
                with default_storage.open(self.firm_logo.name, 'rb') as image_file:
                    image_data = image_file.read()
                    
                # Get the file extension to determine MIME type
                file_extension = self.firm_logo.name.split('.')[-1].lower()
                mime_type_map = {
                    'jpg': 'image/jpeg',
                    'jpeg': 'image/jpeg',
                    'png': 'image/png',
                    'gif': 'image/gif',
                    'webp': 'image/webp'
                }
                mime_type = mime_type_map.get(file_extension, 'image/jpeg')
                
                # Convert to base64 and return as data URL
                base64_data = base64.b64encode(image_data).decode('utf-8')
                return f"data:{mime_type};base64,{base64_data}"
                
            except Exception as e:
                print(f"Error reading logo file: {e}")
                return None
        return None
        

from django.db import models
from django.utils import timezone
import json

class SnowAIMLModelLogEntry(models.Model):
    # Basic model information
    snowai_model_name = models.CharField(max_length=255, db_index=True)
    snowai_model_type = models.CharField(max_length=100, choices=[
        ('classification', 'Classification'),
        ('regression', 'Regression'),
        ('neural_network', 'Neural Network'),
        ('deep_learning', 'Deep Learning'),
        ('ensemble', 'Ensemble'),
        ('time_series', 'Time Series'),
        ('nlp', 'Natural Language Processing'),
        ('computer_vision', 'Computer Vision'),
        ('other', 'Other')
    ])
    snowai_tags = models.TextField(help_text="Comma-separated tags")
    snowai_description = models.TextField(blank=True, null=True)
    
    # Code and implementation
    snowai_code_used = models.TextField(help_text="Full code implementation")
    snowai_colab_notebook_url = models.URLField(blank=True, null=True, help_text="Google Colab notebook URL")
    snowai_framework_used = models.CharField(max_length=100, blank=True, null=True, help_text="e.g., TensorFlow, PyTorch, Scikit-learn")
    
    # Dataset information
    snowai_dataset_name = models.CharField(max_length=255, blank=True, null=True)
    snowai_dataset_description = models.TextField(blank=True, null=True)
    snowai_dataset_size = models.IntegerField(blank=True, null=True, help_text="Number of records")
    snowai_dataset_features = models.IntegerField(blank=True, null=True, help_text="Number of features")
    snowai_dataset_source = models.CharField(max_length=255, blank=True, null=True)
    snowai_financial_market_type = models.CharField(max_length=100, blank=True, null=True, 
                                                   choices=[
                                                       ('stocks', 'Stocks'),
                                                       ('forex', 'Forex'),
                                                       ('crypto', 'Cryptocurrency'),
                                                       ('bonds', 'Bonds'),
                                                       ('commodities', 'Commodities'),
                                                       ('indices', 'Market Indices'),
                                                       ('options', 'Options'),
                                                       ('futures', 'Futures'),
                                                       ('mixed', 'Mixed Markets')
                                                   ])
    
    # Performance metrics
    snowai_accuracy_score = models.FloatField(blank=True, null=True)
    snowai_precision_score = models.FloatField(blank=True, null=True)
    snowai_recall_score = models.FloatField(blank=True, null=True)
    snowai_f1_score = models.FloatField(blank=True, null=True)
    snowai_mae_score = models.FloatField(blank=True, null=True, help_text="Mean Absolute Error")
    snowai_mse_score = models.FloatField(blank=True, null=True, help_text="Mean Squared Error")
    snowai_rmse_score = models.FloatField(blank=True, null=True, help_text="Root Mean Squared Error")
    snowai_r2_score = models.FloatField(blank=True, null=True, help_text="R-squared")
    snowai_auc_score = models.FloatField(blank=True, null=True, help_text="Area Under Curve")
    snowai_custom_metrics = models.JSONField(blank=True, null=True, help_text="Additional custom metrics as JSON")
    
    # Training information
    snowai_training_duration = models.FloatField(blank=True, null=True, help_text="Training time in minutes")
    snowai_epochs_trained = models.IntegerField(blank=True, null=True)
    snowai_batch_size = models.IntegerField(blank=True, null=True)
    snowai_learning_rate = models.FloatField(blank=True, null=True)
    snowai_optimizer_used = models.CharField(max_length=100, blank=True, null=True)
    
    # Financial-specific metrics
    snowai_profit_loss = models.FloatField(blank=True, null=True, help_text="P&L from backtesting")
    snowai_sharpe_ratio = models.FloatField(blank=True, null=True)
    snowai_max_drawdown = models.FloatField(blank=True, null=True)
    snowai_win_rate = models.FloatField(blank=True, null=True, help_text="Percentage of winning trades")
    snowai_roi_percentage = models.FloatField(blank=True, null=True, help_text="Return on Investment %")
    
    # Metadata
    snowai_created_at = models.DateTimeField(default=timezone.now)
    snowai_updated_at = models.DateTimeField(auto_now=True)
    snowai_status = models.CharField(max_length=50, choices=[
        ('experimental', 'Experimental'),
        ('validated', 'Validated'),
        ('production', 'Production'),
        ('deprecated', 'Deprecated')
    ], default='experimental')
    snowai_notes = models.TextField(blank=True, null=True)
    
    class Meta:
        db_table = 'snowai_ml_model_log_entries'
        ordering = ['-snowai_created_at']
        verbose_name = 'SnowAI ML Model Log Entry'
        verbose_name_plural = 'SnowAI ML Model Log Entries'
    
    def __str__(self):
        return f"{self.snowai_model_name} - {self.snowai_model_type}"
    
    @property
    def snowai_tags_list(self):
        """Convert comma-separated tags to list"""
        if self.snowai_tags:
            return [tag.strip() for tag in self.snowai_tags.split(',') if tag.strip()]
        return []
    
    @snowai_tags_list.setter
    def snowai_tags_list(self, tag_list):
        """Set tags from list"""
        self.snowai_tags = ', '.join(tag_list) if tag_list else ''
    
    def snowai_get_primary_metric(self):
        """Get the most relevant metric based on model type"""
        if self.snowai_accuracy_score is not None:
            return {'name': 'Accuracy', 'value': self.snowai_accuracy_score, 'format': '{:.2%}'}
        elif self.snowai_r2_score is not None:
            return {'name': 'R²', 'value': self.snowai_r2_score, 'format': '{:.3f}'}
        elif self.snowai_f1_score is not None:
            return {'name': 'F1', 'value': self.snowai_f1_score, 'format': '{:.3f}'}
        elif self.snowai_mae_score is not None:
            return {'name': 'MAE', 'value': self.snowai_mae_score, 'format': '{:.4f}'}
        return None


class SnowAIModelDatasetFile(models.Model):
    """Store dataset files with persistent URLs to avoid loss on redeployment"""
    snowai_log_entry = models.ForeignKey(SnowAIMLModelLogEntry, on_delete=models.CASCADE, related_name='snowai_dataset_files')
    snowai_file_name = models.CharField(max_length=255)
    snowai_file_type = models.CharField(max_length=50, choices=[
        ('csv', 'CSV'),
        ('json', 'JSON'),
        ('parquet', 'Parquet'),
        ('excel', 'Excel'),
        ('other', 'Other')
    ])
    snowai_file_size = models.IntegerField(help_text="File size in bytes")
    snowai_external_url = models.URLField(help_text="Persistent URL to dataset (e.g., Google Drive, Dropbox)")
    snowai_file_hash = models.CharField(max_length=64, blank=True, null=True, help_text="SHA-256 hash for integrity")
    snowai_upload_date = models.DateTimeField(default=timezone.now)
    snowai_description = models.TextField(blank=True, null=True)
    
    class Meta:
        db_table = 'snowai_model_dataset_files'
        ordering = ['-snowai_upload_date']
    
    def __str__(self):
        return f"{self.snowai_file_name} ({self.snowai_log_entry.snowai_model_name})"


class FingerprintStatus(models.Model):
    user_email = models.EmailField(unique=True, default='tlotlo.motingwe@example.com')
    is_registered = models.BooleanField(default=False)
    domain = models.CharField(max_length=255, blank=True, null=True)
    registered_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user_email} - Registered: {self.is_registered}"



class SnowAITraderHistoryGPTSummary(models.Model):
    summary_text = models.TextField()
    total_trades = models.IntegerField(default=0)
    win_rate = models.FloatField(default=0.0)
    total_profit_loss = models.FloatField(default=0.0)
    best_performing_strategy = models.CharField(max_length=200, blank=True)
    worst_performing_strategy = models.CharField(max_length=200, blank=True)
    most_traded_asset = models.CharField(max_length=100, blank=True)
    average_trade_amount = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Trader History Summary - {self.created_at.strftime('%Y-%m-%d')}"


class SnowAIMacroGPTSummary(models.Model):
    summary_text = models.TextField()
    total_economic_events = models.IntegerField(default=0)
    high_impact_events_count = models.IntegerField(default=0)
    most_active_currency = models.CharField(max_length=10, blank=True)
    key_market_themes = models.TextField(blank=True)
    upcoming_events_preview = models.TextField(blank=True)
    market_sentiment = models.CharField(max_length=50, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Macro Summary - {self.created_at.strftime('%Y-%m-%d')}"


class SnowAIIdeaGPTSummary(models.Model):
    summary_text = models.TextField()
    total_ideas = models.IntegerField(default=0)
    pending_ideas = models.IntegerField(default=0)
    in_progress_ideas = models.IntegerField(default=0)
    completed_ideas = models.IntegerField(default=0)
    most_common_category = models.CharField(max_length=200, blank=True)
    completion_rate = models.FloatField(default=0.0)
    oldest_pending_idea = models.CharField(max_length=500, blank=True)
    newest_idea = models.CharField(max_length=500, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Ideas Summary - {self.created_at.strftime('%Y-%m-%d')}"


class SnowAIBacktestingGPTSummary(models.Model):
    summary_text = models.TextField()
    total_backtests = models.IntegerField(default=0)
    successful_backtests = models.IntegerField(default=0)
    average_sharpe_ratio = models.FloatField(default=0.0)
    average_annual_return = models.FloatField(default=0.0)
    average_max_drawdown = models.FloatField(default=0.0)
    best_performing_strategy = models.CharField(max_length=500, blank=True)
    worst_performing_strategy = models.CharField(max_length=500, blank=True)
    most_used_dataset = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Backtesting Summary - {self.created_at.strftime('%Y-%m-%d')}"


class SnowAIPaperGPTSummary(models.Model):
    summary_text = models.TextField()
    total_papers = models.IntegerField(default=0)
    most_common_category = models.CharField(max_length=200, blank=True)
    total_file_size_mb = models.FloatField(default=0.0)
    average_paper_length = models.IntegerField(default=0)
    latest_upload = models.CharField(max_length=500, blank=True)
    research_recommendations = models.TextField(blank=True)
    key_insights = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Paper Summary - {self.created_at.strftime('%Y-%m-%d')}"


class SnowAIResearchGPTSummary(models.Model):
    summary_text = models.TextField()
    total_research_entries = models.IntegerField(default=0)
    total_papers_analyzed = models.IntegerField(default=0)
    knowledge_gaps_identified = models.TextField(blank=True)
    future_research_directions = models.TextField(blank=True)
    cross_paper_insights = models.TextField(blank=True)
    practical_applications = models.TextField(blank=True)
    research_methodology_suggestions = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Research Summary - {self.created_at.strftime('%Y-%m-%d')}"


class SnowAIConversationHistory(models.Model):
    GPT_CHOICES = [
        ('TraderHistoryGPT', 'TraderHistoryGPT'),
        ('MacroGPT', 'MacroGPT'),
        ('IdeaGPT', 'IdeaGPT'),
        ('BacktestingGPT', 'BacktestingGPT'),
        ('PaperGPT', 'PaperGPT'),
        ('ResearchGPT', 'ResearchGPT'),
    ]
    
    gpt_system = models.CharField(max_length=50, choices=GPT_CHOICES)
    user_message = models.TextField()
    ai_response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.gpt_system} - {self.timestamp.strftime('%Y-%m-%d %H:%M')}"




from django.utils import timezone

class GPTDiscussion(models.Model):
    """Model to store GPT discussions - only one exists at a time"""
    discussion_id = models.CharField(max_length=100, unique=True, default='current_discussion')
    started_at = models.DateTimeField(default=timezone.now)
    completed_at = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=True)
    total_messages = models.IntegerField(default=0)
    central_gpt_summary = models.TextField(blank=True)
    discussion_metrics = models.JSONField(default=dict)
    trigger_type = models.CharField(max_length=20, choices=[
        ('scheduled', 'Scheduled'),
        ('manual', 'Manual')
    ], default='scheduled')
    
    class Meta:
        ordering = ['-started_at']

class GPTDiscussionMessage(models.Model):
    """Individual messages in the GPT discussion"""
    discussion = models.ForeignKey(GPTDiscussion, on_delete=models.CASCADE, related_name='messages')
    gpt_system = models.CharField(max_length=50, choices=[
        ('TraderHistoryGPT', 'TraderHistoryGPT'),
        ('MacroGPT', 'MacroGPT'),
        ('IdeaGPT', 'IdeaGPT'),
        ('BacktestingGPT', 'BacktestingGPT'),
        ('PaperGPT', 'PaperGPT'),
        ('ResearchGPT', 'ResearchGPT'),
        ('CentralGPT', 'CentralGPT')
    ])
    message = models.TextField()
    timestamp = models.DateTimeField(default=timezone.now)
    turn_number = models.IntegerField(default=1)
    
    class Meta:
        ordering = ['timestamp']



from django.utils import timezone

class SnowAIVideoTranscriptRecord(models.Model):
    # Core identification fields
    transcript_uuid = models.CharField(max_length=100, unique=True, db_index=True)
    youtube_video_id = models.CharField(max_length=50, blank=True, null=True)
    youtube_url = models.URLField(max_length=500, blank=True, null=True)
    video_title = models.CharField(max_length=300, blank=True, null=True)
    
    # Speaker and context information
    primary_speaker_name = models.CharField(max_length=200, blank=True, null=True)
    speaker_organization = models.CharField(max_length=200, blank=True, null=True)
    speaker_country_code = models.CharField(max_length=10, blank=True, null=True)
    speaker_country_name = models.CharField(max_length=100, blank=True, null=True)
    
    # Content and metadata
    full_transcript_text = models.TextField()
    video_duration_seconds = models.IntegerField(blank=True, null=True)
    transcript_language = models.CharField(max_length=10, default='en')
    video_upload_date = models.DateTimeField(blank=True, null=True)
    
    # Processing metadata
    transcription_method = models.CharField(max_length=50, default='youtube_auto')  # youtube_auto, manual, ai_generated
    transcript_confidence_score = models.FloatField(blank=True, null=True)
    processing_status = models.CharField(max_length=30, default='completed')
    
    # Search and categorization
    content_category = models.CharField(max_length=100, blank=True, null=True)  # central_bank, government, corporate
    economic_topics = models.JSONField(default=list, blank=True)  # ["monetary_policy", "inflation", "interest_rates"]
    custom_tags = models.JSONField(default=list, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    archived_at = models.DateTimeField(blank=True, null=True)
    
    # Additional analysis fields
    word_count = models.IntegerField(default=0)
    sentiment_analysis_score = models.FloatField(blank=True, null=True)
    key_phrases_extracted = models.JSONField(default=list, blank=True)
    
    class Meta:
        db_table = 'snowai_video_transcript_records'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['primary_speaker_name']),
            models.Index(fields=['speaker_country_code']),
            models.Index(fields=['content_category']),
            models.Index(fields=['created_at']),
            models.Index(fields=['video_upload_date']),
        ]

    def __str__(self):
        return f"{self.primary_speaker_name or 'Unknown'} - {self.video_title[:50] or 'Untitled'}..."

    def save(self, *args, **kwargs):
        if self.full_transcript_text:
            self.word_count = len(self.full_transcript_text.split())
        super().save(*args, **kwargs)


class SnowAITranscriptSearchHistory(models.Model):
    search_query = models.CharField(max_length=500)
    search_filters = models.JSONField(default=dict)
    results_count = models.IntegerField(default=0)
    search_timestamp = models.DateTimeField(auto_now_add=True)
    user_session_id = models.CharField(max_length=100, blank=True, null=True)
    
    class Meta:
        db_table = 'snowai_transcript_search_history'
        ordering = ['-search_timestamp']



# models.py (add this to your existing models)

class SnowAITranscriptAnalysis(models.Model):
    # Foreign key to the transcript
    transcript = models.OneToOneField(
        SnowAIVideoTranscriptRecord, 
        on_delete=models.CASCADE, 
        related_name='ai_analysis'
    )
    
    # Analysis UUID for unique identification
    analysis_uuid = models.CharField(max_length=100, unique=True, db_index=True)
    
    # Main Analysis Results
    executive_summary = models.TextField()
    key_themes = models.JSONField(default=list)  # ["monetary_policy", "inflation_outlook", "economic_growth"]
    
    # Economic Insights
    economic_opportunities = models.JSONField(default=list)  # [{"opportunity": "text", "confidence": 0.8}]
    economic_risks = models.JSONField(default=list)  # [{"risk": "text", "impact_level": "high"}]
    policy_implications = models.JSONField(default=list)  # [{"implication": "text", "timeframe": "short_term"}]
    
    # Market Sentiment Analysis
    overall_sentiment = models.CharField(max_length=20)  # positive, negative, neutral, mixed
    sentiment_confidence = models.FloatField(default=0.0)
    market_outlook = models.CharField(max_length=20)  # bullish, bearish, neutral, uncertain
    
    # Key Metrics Mentioned
    inflation_mentions = models.JSONField(default=dict)  # {"current": "3.2%", "target": "2%", "forecast": "2.8%"}
    interest_rate_mentions = models.JSONField(default=dict)  # {"current": "5.25%", "next_meeting": "hold"}
    gdp_mentions = models.JSONField(default=dict)  # {"current": "2.1%", "forecast": "1.8%"}
    unemployment_mentions = models.JSONField(default=dict)  # {"current": "3.8%", "forecast": "4.1%"}
    
    # Action Items and Predictions
    policy_actions_suggested = models.JSONField(default=list)
    market_predictions = models.JSONField(default=list)  # [{"prediction": "text", "timeframe": "6_months", "confidence": 0.7}]
    
    # Analysis Metadata
    analysis_model_used = models.CharField(max_length=50, default='gpt-4o-mini')
    analysis_prompt_version = models.CharField(max_length=20, default='v1.0')
    analysis_duration_seconds = models.FloatField(blank=True, null=True)
    analysis_word_count = models.IntegerField(default=0)
    
    # Timestamps
    analysis_created_at = models.DateTimeField(auto_now_add=True)
    analysis_updated_at = models.DateTimeField(auto_now=True)
    
    # Quality Metrics
    analysis_completeness_score = models.FloatField(default=0.0)  # 0-1 score
    key_insights_count = models.IntegerField(default=0)
    
    class Meta:
        db_table = 'snowai_transcript_analysis'
        ordering = ['-analysis_created_at']
        indexes = [
            models.Index(fields=['overall_sentiment']),
            models.Index(fields=['market_outlook']),
            models.Index(fields=['analysis_created_at']),
        ]

    def __str__(self):
        return f"Analysis for {self.transcript.primary_speaker_name or 'Unknown'} - {self.overall_sentiment}"

    def save(self, *args, **kwargs):
        # Auto-generate UUID if not provided
        if not self.analysis_uuid:
            import uuid
            self.analysis_uuid = str(uuid.uuid4())
        
        # Count key insights
        self.key_insights_count = (
            len(self.economic_opportunities) + 
            len(self.economic_risks) + 
            len(self.policy_implications) +
            len(self.market_predictions)
        )
        
        super().save(*args, **kwargs)



class SnowAIHedgeFundEntity(models.Model):
    name = models.CharField(max_length=255)
    logo_url = models.URLField(max_length=500, blank=True, null=True, help_text="External URL for logo (e.g., imgur, cloudinary)")
    description = models.TextField(blank=True, null=True)
    founded_year = models.IntegerField(blank=True, null=True)
    aum = models.CharField(max_length=100, blank=True, null=True, help_text="Assets Under Management")
    strategy = models.CharField(max_length=255, blank=True, null=True)
    headquarters = models.CharField(max_length=255, blank=True, null=True)
    website = models.URLField(max_length=500, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "SnowAI Hedge Fund Entity"
        verbose_name_plural = "SnowAI Hedge Fund Entities"
    
    def __str__(self):
        return self.name


class SnowAIHedgeFundKeyPerson(models.Model):
    hedge_fund = models.ForeignKey(SnowAIHedgeFundEntity, on_delete=models.CASCADE, related_name='key_people')
    name = models.CharField(max_length=255)
    role = models.CharField(max_length=255, blank=True, null=True)
    wikipedia_url = models.URLField(max_length=500, blank=True, null=True)
    linkedin_url = models.URLField(max_length=500, blank=True, null=True)
    bio = models.TextField(blank=True, null=True)
    photo_url = models.URLField(max_length=500, blank=True, null=True, help_text="External URL for photo")
    
    class Meta:
        ordering = ['name']
        verbose_name = "SnowAI Hedge Fund Key Person"
        verbose_name_plural = "SnowAI Hedge Fund Key People"
    
    def __str__(self):
        return f"{self.name} - {self.hedge_fund.name}"


class SnowAIHedgeFundResource(models.Model):
    hedge_fund = models.ForeignKey(SnowAIHedgeFundEntity, on_delete=models.CASCADE, related_name='resources')
    title = models.CharField(max_length=255)
    url = models.URLField(max_length=500)
    description = models.TextField(blank=True, null=True)
    resource_type = models.CharField(max_length=50, choices=[
        ('article', 'Article'),
        ('interview', 'Interview'),
        ('video', 'Video'),
        ('report', 'Report'),
        ('other', 'Other')
    ], default='article')
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "SnowAI Hedge Fund Resource"
        verbose_name_plural = "SnowAI Hedge Fund Resources"
    
    def __str__(self):
        return f"{self.title} - {self.hedge_fund.name}"


class SnowAIHedgeFundPerformance(models.Model):
    hedge_fund = models.ForeignKey(SnowAIHedgeFundEntity, on_delete=models.CASCADE, related_name='performance_data')
    year = models.IntegerField()
    return_percentage = models.DecimalField(max_digits=10, decimal_places=2, help_text="Annual return percentage")
    notes = models.TextField(blank=True, null=True)
    
    class Meta:
        ordering = ['year']
        unique_together = ['hedge_fund', 'year']
        verbose_name = "SnowAI Hedge Fund Performance"
        verbose_name_plural = "SnowAI Hedge Fund Performance Data"
    
    def __str__(self):
        return f"{self.hedge_fund.name} - {self.year}: {self.return_percentage}%"


class AssetBiasRecommendation(models.Model):
    """
    Store AI-generated bias recommendations for trading assets
    One entry per asset - automatically updates on save
    """
    BIAS_CHOICES = [
        ('bullish', 'Bullish'),
        ('bearish', 'Bearish'),
        ('neutral', 'Neutral'),
    ]
    
    VOLUME_CHOICES = [
        ('high', 'High'),
        ('medium', 'Medium'),
        ('low', 'Low'),
    ]
    
    # Core fields only
    asset_name = models.CharField(max_length=100, unique=True, db_index=True)
    bias = models.CharField(max_length=20, choices=BIAS_CHOICES, null=True, blank=True)
    volume = models.CharField(max_length=20, choices=VOLUME_CHOICES, null=True, blank=True)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-updated_at']
        verbose_name = 'Asset Bias Recommendation'
        verbose_name_plural = 'Asset Bias Recommendations'
    
    def __str__(self):
        return f"{self.asset_name} - Bias: {self.bias or 'None'}, Volume: {self.volume or 'None'}"



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