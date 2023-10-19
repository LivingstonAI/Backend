from django.core.exceptions import ObjectDoesNotExist
from django.shortcuts import get_object_or_404
from django.http import JsonResponse
from django.contrib.auth import authenticate, login
from django.views.decorators.csrf import csrf_exempt
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import *
from .models import *
from django.core import serializers
from django.utils import timezone
from collections import Counter, defaultdict
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
import datetime
from datetime import date
import time
import os
import http.client
import urllib.parse
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, resample_apply
import asyncio
from backtesting.test import SMA, GOOG, EURUSD
import pandas as pd
import pandas_ta as ta

current_hour = datetime.datetime.now().time().hour


def get_openai_key(request):
    return JsonResponse({'OPENAI_API_KEY': os.environ['OPENAI_API_KEY']})


def get_news_data():
    # Get the current date
    news_data = []
    current_date = datetime.datetime.now().date()
    day_abbreviation = current_date.strftime('%a')
    day_encountered = False
    all_days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    next_day = current_date + datetime.timedelta(days=1)
    next_day_found = False
    # Format the next day's date to get the abbreviated day name
    next_day_abbreviated = next_day.strftime('%a')
    try:
        # Check if there's existing data for the current date
        existing_news_data = NewsData.objects.filter(created_at__date=current_date)
        
        if existing_news_data.exists():
            # Fetch and return the existing news data
            existing_data = existing_news_data.first().data
            return existing_data

        chrome_options = Options()
        chrome_options.add_argument("--headless")
        # chrome_options.headless = True
        service = Service(executable_path='snowAIWeb/chromedriver.exe')
        driver = webdriver.Chrome(service=service)
        driver.get("http://www.forexfactory.com")

        table = driver.find_element(By.CLASS_NAME, "calendar__table")
        # Iterate over each table row
        for row in table.find_elements(By.TAG_NAME, "tr"):
            # list comprehension to get each cell's data and filter out empty cells
            row_data = list(filter(None, [td.text for td in row.find_elements(By.TAG_NAME, "td")]))
            if row_data == []:
                continue
            if len(row_data) > 1:
                if next_day_abbreviated in row_data[0]:
                    next_day_found = True
                if next_day_found:
                    break
                if day_abbreviation in row_data[0] and not next_day_found:
                    day_encountered = True
                if day_encountered:
                    news_data.append(row_data)

        # Save news data to the model
        news_model = NewsData(data=news_data, created_at=current_date)
        news_model.save()

        # driver.close()
        driver.quit()
        return news_data
    except Exception as e:
        print(f'Error: {e}')



class UserRegistrationView(APIView):
    @csrf_exempt
    def post(self, request, format=None):
        serializer = UserRegistrationSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@csrf_exempt
def check_email(request):
    if request.method == "POST":
        email = request.POST.get("email")
        existing_user = User.objects.filter(email=email).exists()
        return JsonResponse({"exists": existing_user})


@csrf_exempt
def all_trades(request, email):
    trades = Trade.objects.filter(email=email)
    serialized_trades = serializers.serialize('json', trades)
    return JsonResponse({'trades': serialized_trades})


class TellUsMoreCreateView(APIView):
    def post(self, request, *args, **kwargs):
        data = request.data
        data['tell_us_more_user'] = request.user.id  # Add the logged-in user's primary key
        # Create a serializer instance
        serializer = TellUsMoreSerializer(data=data)
        # Check if serializer is valid and print any errors
        if serializer.is_valid():
            # Save the serializer instance
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class UserLoginView(APIView):
    @csrf_exempt
    def post(self, request, *args, **kwargs):
        global email_of_user
        email = request.data.get('email')
        password = request.data.get('password')
        test_user = User.objects.get(email=email)
    
        user = authenticate(request, email=email, password=password)
        
        if user is not None:
            # request.session['USER_EMAIL'] = email  # Store email in session
            email_of_user = email
            login(request,user)
            return Response({'message': 'Login successful', 'email': email}, status=status.HTTP_200_OK)
        else:
            return Response({'message': 'Invalid login credentials'}, status=status.HTTP_401_UNAUTHORIZED)




class TradeView(APIView):
    def post(self, request, *args, **kwargs):
        # email = request.data.get('email')
        data = request.data
        email = data['email']
        initial_capital = TellUsMore.objects.get(user_email=email).initial_capital
        asset = data['asset']
        order_type = data['order_type']
        strategy = data['strategy']
        position_size = data['position_size']
        timeframe = data['timeframe']
        entry_date = data['start_date']
        exit_date = data['end_date']
        entry_point = data['entry_point']
        stop_loss = data['stop_loss']
        take_profit = data['take_profit']
        exit_point = data['exit_point']
        outcome = data['outcome']
        amount = data['amount']
        emotional_bias = data['emotional_bias']
        reflection = data['reflection']
        if outcome == 'Loss':
            amount = -amount
        roi = (amount/initial_capital) * 100
        new_trade = Trade(
            email = email,
            asset=asset,
            order_type=order_type,
            strategy=strategy,
            lot_size=position_size,
            roi=roi,
            timeframe=timeframe,
            entry_date=entry_date,
            exit_date=exit_date,
            entry_point=entry_point,
            stop_loss=stop_loss,
            take_profit=take_profit,
            exit_point=exit_point,
            outcome=outcome,
            amount=amount,
            emotional_bias=emotional_bias,
            reflection=reflection
        )
        new_trade.save()
        return Response({''}, status=status.HTTP_200_OK)


@csrf_exempt
def full_trade(request, trade_id):
    try:
        trade = Trade.objects.get(pk=trade_id)
        trade_data = {
            "model": "snowAIWeb.trade",
            "pk": trade_id,
            "fields": {
                "email": trade.email,
                "asset": trade.asset,
                "order_type": trade.order_type,
                "strategy": trade.strategy,
                "lot_size": trade.lot_size,
                "timeframe": trade.timeframe,
                "entry_date": trade.entry_date,
                "exit_date": trade.exit_date,
                "entry_point": trade.entry_point,
                "stop_loss": trade.stop_loss,
                "take_profit": trade.take_profit,
                "exit_point": trade.exit_point,
                "outcome": trade.outcome,
                "amount": trade.amount,
                "emotional_bias": trade.emotional_bias,
                "reflection": trade.reflection,
                'roi': trade.roi
            }
        }
        return JsonResponse({"trade": trade_data})
    except Trade.DoesNotExist:
        return JsonResponse({"error": "Trade not found"}, status=404)


@csrf_exempt
def user_overview(request, user_email):
    # Journal.objects.all().delete()
    initial_capital = TellUsMore.objects.get(user_email=user_email).initial_capital
    trades = Trade.objects.filter(email=user_email)
    print(f'User is {request.user}')
    
    total_profit = sum(trade.amount for trade in trades)
    equity_amount = initial_capital + total_profit
    roi = (total_profit / initial_capital) * 100 if initial_capital > 0 else 0

    equity_values = [initial_capital]
    for trade in trades:
        equity_values.append(equity_values[-1] + trade.amount)

    peak_value = max(equity_values)
    trough_value = min(equity_values)

    if peak_value > 0:
        maximum_drawdown = ((peak_value - trough_value) / peak_value) * 100
    else:
        maximum_drawdown = 0

    total_trades = Trade.objects.filter(email=user_email).count()
    winning_trades = Trade.objects.filter(email=user_email,outcome='Profit').count()
    losing_trades = Trade.objects.filter(email=user_email, outcome='Loss').count()

    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    loss_rate = (losing_trades / total_trades) * 100 if total_trades > 0 else 0

    if total_trades > 0:
        risk_of_ruin = (losing_trades / total_trades) * 100
    else:
        risk_of_ruin = 0

     # Calculate net profit for each strategy
    strategy_net_profit = {}  # Dictionary to store strategy net profit
    for trade in trades:
        strategy = trade.strategy
        if strategy in strategy_net_profit:
            strategy_net_profit[strategy] += trade.amount
        else:
            strategy_net_profit[strategy] = trade.amount

    # Find the best and worst strategies based on net profit
    best_strategy = max(strategy_net_profit, key=strategy_net_profit.get, default="N/A")
    worst_strategy = min(strategy_net_profit, key=strategy_net_profit.get, default="N/A")

    timeframe_net_profit = {}  # Dictionary to store timeframe net profit
    for trade in trades:
        timeframe = trade.timeframe
        if timeframe in timeframe_net_profit:
            timeframe_net_profit[timeframe] += trade.amount
        else:
            timeframe_net_profit[timeframe] = trade.amount

    # Find the best and worst timeframes based on net profit
    best_timeframe = max(timeframe_net_profit, key=timeframe_net_profit.get, default="N/A")
    worst_timeframe = min(timeframe_net_profit, key=timeframe_net_profit.get, default="N/A")

    current_year = timezone.now().year
     # Calculate the day of most wins and day of most losses
    days_of_wins = [trade.entry_date.weekday() for trade in trades if trade.outcome == 'Profit']
    days_of_losses = [trade.entry_date.weekday() for trade in trades if trade.outcome == 'Loss']

    most_common_day_of_wins = Counter(days_of_wins).most_common(1)
    most_common_day_of_losses = Counter(days_of_losses).most_common(1)

    try:
        day_of_most_losses = datetime.date(current_year, 1, most_common_day_of_losses[0][0] + 1).strftime("%A")
    except:
        day_of_most_losses = 'None'
    try:
        day_of_most_wins = datetime.date(current_year, 1, most_common_day_of_losses[0][0] + 1).strftime("%A")
    except:
        day_of_most_wins = 'None'

    equity_dates = [trade.entry_date.date().strftime('%Y-%m-%d') for trade in trades]
    equity_dates.sort()
    equity_values = []
    true_amount = 0
    current_capital = initial_capital
    for trade in trades:
        equity_values.append(current_capital+trade.amount)
        current_capital += trade.amount
    equity_labels = [f"Trade{i + 1}" for i in range(len(trades))]
    
    return JsonResponse({
        'equity_amount': equity_amount,
        'profit': total_profit,
        'roi': round(roi, 2),
        'maximum_drawdown': maximum_drawdown,
        'risk_of_ruin': risk_of_ruin,
        'win_rate': win_rate,
        'loss_rate': loss_rate,
        'best_strategy': best_strategy,
        'worst_strategy': worst_strategy,
        'best_timeframe': best_timeframe,
        'worst_timeframe': worst_timeframe,
        'day_of_most_wins': day_of_most_wins,
        'day_of_most_losses': day_of_most_losses,
        'equity_over_time_data': equity_values,
        'equity_over_time_labels': equity_labels,
    })


@csrf_exempt
def save_journal(request, user_email):
    if request.method == 'POST':
        current_date = datetime.datetime.now()
        try:
            data = json.loads(request.body)
            content = data.get('content', '')
            tags = data.get('tags', '')  # Extract tags from JSON data

            if content:
                journal = Journals(user_email=user_email, content=content, created_date=current_date, tags=tags)
                journal.save()
                return JsonResponse({'message': 'Journal entry saved successfully.'})
            else:
                return JsonResponse({'error': 'Content cannot be empty.'}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data.'}, status=400)
    else:
        return JsonResponse({'error': 'Invalid request method.'}, status=405)


@csrf_exempt
def fetch_journals(request, user_email):
    journals = Journals.objects.filter(user_email=user_email)
    journal_data = []

    for journal in journals:
        journal_data.append({
            'id': journal.id,
            'content': journal.content,
            'created_date': journal.created_date.strftime('%Y/%m/%d: %H:%M'),
            'tags': journal.tags,
        })
    return JsonResponse({'journals': journal_data})


@csrf_exempt
def view_journal(request, journal_id):
    journal = Journals.objects.get(pk=journal_id)
    journal_data = {
        'id': journal.id,
        'content': journal.content,
        'created_date': journal.created_date.strftime('%Y/%m/%d %H:%M'),
        'tags': journal.tags,
    }
    return JsonResponse({'journal': journal_data})


@csrf_exempt
def upcoming_news(request, user_email):
    try:
        user_assets = TellUsMore.objects.get(user_email=user_email).main_assets
        data = NewsData.objects.all()
        current_date = datetime.datetime.now().date()
        # Assuming news_data_queryset contains the queryset of NewsData objects
        all_dates = [extract_date(news_obj.created_at) for news_obj in data]
        # Check if there is any NewsData with the given current_date
        if not NewsData.objects.filter(created_at__date=current_date).exists():
            news_data = get_news_data()  # Fetch news data
        else:
            news_data = NewsData.objects.filter(created_at__date=current_date).first().data
        news_data = clean_news_data(news_data)
        return JsonResponse(news_data)
    except ObjectDoesNotExist:
        return JsonResponse({'message': 'User not found'}, status=404)
    except Exception as e:
        print(f'Error: {e}')
        return JsonResponse({'message': str(e)}, status=500)


@csrf_exempt
def extract_date(date):
    timestamp_string = str(date)
    timestamp = datetime.datetime.strptime(timestamp_string, "%Y-%m-%d %H:%M:%S.%f%z")
    date = timestamp.date()
    return date


# Sample news data
@csrf_exempt
def extract_time(news):
    time_pattern = r'(\d{1,2}:\d{2}(?:am|pm))'
    for item in news:
        match = re.search(time_pattern, item)
        if match:
            return match.group(1)
    return None


def clean_news_data(news_data):
    final_dict = {}
    date_time = ''
    for data in news_data:
        for more_data in data:
            if 'am' in more_data or 'pm' in more_data:
                final_dict[more_data] = []
                date_time = more_data
        try:
            final_dict[date_time].append(data)
        except Exception as e:
            print(f'Error: {e}')
    return final_dict


@csrf_exempt
def get_user_data(request, user_email):
    print(request.user)
    try:
        user_data = TellUsMore.objects.get(user_email=user_email)
        data = {
            "user_email": user_data.user_email,
            "trading_experience": user_data.trading_experience,
            "main_assets": user_data.main_assets,
            "initial_capital": user_data.initial_capital,
            "trading_goals": user_data.trading_goals,
            "benefits": user_data.benefits,
        }
        return JsonResponse(data)
    except TellUsMore.DoesNotExist:
        return JsonResponse({"error": "User data not found"}, status=404)



@csrf_exempt
def save_conversation(request, user_email, identifier):
    if request.method == 'POST':
        conversation_data = request.POST.get('conversation_data')  # Assuming you send the data as a POST request
        # Or if you're using JSON in the request body: conversation_data = request.body['conversation_data']
        
        # Check if a conversation with the given identifier already exists
        existing_conversation = Conversation.objects.filter(user_email=user_email, conversation_id=identifier).first()
        if existing_conversation:
            # Update the existing conversation with new conversation data
            existing_conversation.conversation = conversation_data
            existing_conversation.save()
            return JsonResponse({'message': 'Conversation data updated successfully'})
        else:
            # Create a new conversation
            conversation = Conversation.objects.create(user_email=user_email, conversation=conversation_data, conversation_id=identifier)
            return JsonResponse({'message': 'Conversation data saved successfully'})
    else:
        return JsonResponse({'error': 'Invalid request method'})


@csrf_exempt
def fetch_conversations(request, user_email):
    all_conversations = Conversation.objects.filter(user_email=user_email)
    conversations_data = [{'id': conversation.conversation_id, 'conversation': conversation.conversation} for conversation in all_conversations]
    return JsonResponse({'conversations': conversations_data})


@csrf_exempt
def fetch_conversation(request, conversation_id):
    print(request.user)
    try:
        conversation = Conversation.objects.filter(conversation_id=conversation_id).first()
        conversation_data = {'id': conversation.conversation_id, 'conversation': conversation.conversation}
        return JsonResponse({'conversations': conversation_data})
    except Exception as e:
        return JsonResponse({'error': f'error occured in fetch converstion func, it is {e}'})


@csrf_exempt
def update_conversation(request, conversation_id):
    if request.method == 'POST':
        conversation = Conversation.objects.get(conversation_id=conversation_id)
        
        # Update the conversation data as needed
        # For example, if you want to update the conversation content
        new_conversation_data = request.POST.get('new_conversation_data')
        conversation.conversation = new_conversation_data
        
        # Save the updated conversation object
        conversation.save()
        
        return JsonResponse({'message': 'Conversation updated successfully'})
    else:
        return JsonResponse({'error': 'Only POST requests are allowed for updating conversations'})


@csrf_exempt
def delete_conversation(request, conversation_id):
    if request.method == 'POST':
        conversation = Conversation.objects.filter(conversation_id=conversation_id).first()
        conversation.delete()
        return JsonResponse({'message': 'Conversation Successfully deleted'})
    else:
        return JsonResponse({'error': 'Could not delete successfully'})


@csrf_exempt
def update_tell_us_more(request, user_email):
    try:
        # Retrieve the existing user data
        current_user_data = TellUsMore.objects.filter(user_email=user_email).first()

        # Extract the new data from the request
        data = json.loads(request.body)

        # Update the existing user data with the new data
        current_user_data.trading_experience = data["trading_experience"]
        current_user_data.main_assets = data["main_assets"]
        current_user_data.initial_capital = data["initial_capital"]
        current_user_data.trading_goals = data["trading_goals"]
        current_user_data.benefits = data["benefits"]

        # Save the updated data
        current_user_data.save()

        return JsonResponse({"message": "Data updated successfully."}, status=200)
    except TellUsMore.DoesNotExist:
        return JsonResponse({"error": "User data not found."}, status=404)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    

@csrf_exempt
def update_user_assets(request, user_email):
    if request.method == 'POST':
        try:
            # Retrieve the user's TellUsMore instance
            user_assets = TellUsMore.objects.get(user_email=user_email)

            # Parse the request data if it contains new assets
            request_data = json.loads(request.body)
            new_assets = request_data.get('new_assets', None)
            print(new_assets)

            if new_assets is not None:
                # Update the user's main_assets field with the new assets
                user_assets.main_assets = new_assets
                user_assets.save()

                return JsonResponse({'message': 'User assets updated successfully'})
            else:
                return JsonResponse({'error': 'No new assets provided'}, status=400)
        except TellUsMore.DoesNotExist:
            return JsonResponse({'error': 'User not found'}, status=404)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)  


@csrf_exempt
def save_news_data(request):
    # List of assets to fetch news data for
    assets_to_fetch = [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "NZDUSD", "USDCAD",
        "EURGBP", "AUDCAD", "EURZAR", "Gold",  # Add more assets as needed
    ]

    # Establish a connection to the API
    conn = http.client.HTTPSConnection('api.marketaux.com')

    # Define query parameters
    params_template = {
        'api_token': 'xH2KZ1sYqHmNRpfBVfb9C1BbItHMtlRIdZQoRlYw',
        'langauge': 'en',
        'limit': 50,
    }

    # Iterate through the assets and make API requests
    for asset in assets_to_fetch:
        # Update the symbol in the query parameters
        params = params_template.copy()
        params['symbols'] = asset

        # Send a GET request
        conn.request('GET', '/v1/news/all?{}'.format(urllib.parse.urlencode(params)))

        # Get the response
        res = conn.getresponse()

        # Read the response data
        data = res.read()

        # Decode the data from bytes to a string
        data_str = data.decode('utf-8')

        # Parse the JSON data
        news_data = json.loads(data_str)

        # Iterate through the news articles and save specific fields to the database
        for article in news_data['data']:
            title = article['title']
            description = article['description']
            source = article['source']
            url = article['url']
            highlights = article['entities'][0]['highlights'] if article.get('entities') else ''

            # Create a dictionary with the specific fields
            news_entry_data = {
                'title': title,
                'description': description,
                'source': source,
                'url': url,
                'highlights': highlights,
            }

            # Create a News instance and save it to the database
            news_entry = News(
                symbol=asset,  # Set the symbol to the current asset
                data=json.dumps(news_entry_data),  # Store the specific fields as JSON data
                day_created=datetime.datetime.now(),  # Use the current datetime as the day_created value
            )
            news_entry.save()
    return JsonResponse({'message': 'News data saved successfully.'})


@csrf_exempt
def fetch_news_data(request):
    # Fetch all news data without using serializers
    news_objects = News.objects.all()
    
    # Create a list of dictionaries representing the model instances
    # news_data = [
    #     {
    #         "title": news.title,
    #         "description": news.description,
    #         "source": news.source,
    #         # Add more fields as needed
    #         # "highlights": news['entities'][0]['highlights']
    #     }
    #     for news in news_objects
    # ]

    news_data = []
    for news in news_objects:
        news_data.append({
            "symbol": news.symbol,
            "description": news.data,
            "created_on": news.day_created
        })
    
    # Convert the list to JSON and return it
    return JsonResponse({"news_data": news_data}, safe=False)


@csrf_exempt
async def handle_api_request(type_1, type_2, ma1, ma2):
    df_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), './XAUUSD.csv')
    df = pd.read_csv(df_path).drop_duplicates()
    df.index = pd.to_datetime(df['Time'].values)
    del df['Time']
    bt = Backtest(df, SmaCross, exclusive_orders=False, cash=10000)
    output = await bt.run()
    return output


@csrf_exempt
async def moving_average_bot(request, type_1, type_2, ma1, ma2):
    # Use asyncio.gather to run the asynchronous function concurrently
    results = await handle_api_request(type_1, type_2, ma1, ma2) 

    # Process the results and return the response
    return JsonResponse({'Output': await results})


# https://backend-production-c0ab.up.railway.app/create-bot/sma/ema/200/50
class SmaCross(Strategy):
    n0 = 18 # Exponential Moving Average
    n1 = 50 # Exponential Moving Average
    n2 = 200 # Simple Moving Average
    equity = 100000
    risk_percentage = 5
    reward_percentage = 100
    # current_price = 0
    reward_ratio = 15
    position_size = 0.01
    current_position = ''
    range = 2


    def init(self):
        price = self.data.Close
        self.ma1 = self.I(SMA, price, 50)
        self.ma2 = self.I(SMA, price, 200)
        close = self.data.Close


    def check_moving_averages_for_buy(self, df, range):
      past_10_rows = df[['EMA_50', 'SMA_200']].tail(range)
      past_10_rows['Converge'] = past_10_rows['EMA_50'] < past_10_rows['SMA_200']
      past = past_10_rows.tail(1)['Converge'].values[0]
      second_last_row = past_10_rows['Converge'].iloc[-2]
      if past == False and second_last_row == True:
        # print('True')
        return True
      else:
        # print('False')
        return False


    def check_moving_averages_for_sell(self, df, range):
      past_10_rows = df[['EMA_50', 'SMA_200']].tail(range)
      past_10_rows['Diverge'] = past_10_rows['EMA_50'] > past_10_rows['SMA_200']
      past = past_10_rows.tail(1)['Diverge'].values[0]
      second_last_row = past_10_rows['Diverge'].iloc[-2]
      # print(past)
      if past == False and second_last_row == True:
        # print('True')
        return True
      else:
        # print('False')
        return False


    def moving_average(self, df):
        if df.tail(1)['EMA_50'].values[0] > df.tail(1)['SMA_200'].values[0]:
            # price = self.data.Close[-1]
            # gain_amount = self.reward_percentage * self.equity
            # risk_amount = self.risk_percentage * self.equity
            # tp_level = price + (gain_amount/self.equity)
            # sl_level = price - (risk_amount/self.equity)
            # if self.position:
            #   self.position.close()
            tp_level = self.data.Close[-1] + self.reward_percentage
            sl_level = self.data.Close[-1] - self.risk_percentage
            if self.check_moving_averages_for_buy(df=df, range=self.range):
              if self.position:
                self.position.close()
              self.buy(sl=sl_level, tp=tp_level)
        elif df.tail(1)['EMA_50'].values[0] < df.tail(1)['SMA_200'].values[0]:
            # price = self.data.Close[-1]
            # gain_amount = self.reward_percentage * self.equity
            # risk_amount = self.risk_percentage * self.equity
            # tp_level = price - (gain_amount/self.equity)
            # sl_level = price + (risk_amount/self.equity)
            # if self.position:
            #   self.position.close()
            # if self.current_position != 'sell':
            tp_level = self.data.Close[-1] - self.reward_percentage
            sl_level = self.data.Close[-1] + self.risk_percentage
            if self.check_moving_averages_for_sell(df=df, range=self.range):
              if self.position:
                self.position.close()
              self.sell(sl=sl_level, tp=tp_level)


    def next(self):
        df = pd.DataFrame({'Open': self.data.Open, 'High': self.data.High, 'Low': self.data.Low, 'Close': self.data.Close, 'Volume': self.data.Volume})
        # df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df["SMA_200"] = ta.sma(df["Close"], length=200)
        df["EMA_50"] = ta.ema(df["Close"], length=50)
        df["EMA_18"] = ta.ema(df["Close"], length=18)
        df["EMA_6"] = ta.ema(df["Close"], length=6)
        try:
          self.moving_average(df)
          # self.check_moving_averages_for_buy(df, self.range)
        except Exception as e:
          print(f'Error occured: {e}')
          pass



