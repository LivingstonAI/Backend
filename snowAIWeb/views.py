from django.core.exceptions import ObjectDoesNotExist
from django.shortcuts import get_object_or_404
from django.http import JsonResponse, HttpResponse, HttpResponseNotFound
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
from datetime import datetime, timedelta
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
# import MetaTrader5 as mt
from datetime import datetime
# from matplotlib import pyplot as plt
import pandas_ta as ta
import yfinance as yf
import base64
import requests
# Comment
# current_hour = datetime.datetime.now().time().hour


def is_bullish_run(candle1, candle2, candle3, candle4):
    if candle2.Close > candle1.Close and candle3.Close > candle2.Close and candle4.Close > candle3.Close:
        return True
    return False


def is_bearish_run(candle1, candle2, candle3, candle4):
    if candle2.Close < candle1.Close and candle3.Close < candle2.Close and candle4.Close < candle3.Close:
        return True
    return False


def is_bullish_run_3(candle1, candle2, candle3):
    if candle2.Close > candle1.Close and candle3.Close > candle2.Close:
        return True
    return False


def is_bearish_run_3(candle1, candle2, candle3):
    if candle2.Close < candle1.Close and candle3.Close < candle2.Close:
        return True
    return False


def is_bearish_candle(candle):
    if candle.Close < candle.Open:
        return True
    return False


def is_bullish_candle(candle):
    if candle.Close > candle.Open:
        return True
    return False


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
    all_conversations = Conversation.objects.filter(user_email=user_email).order_by('-id')
    conversations_data = [{'id': conversation.conversation_id, 'conversation': conversation.conversation} for conversation in all_conversations]
    return JsonResponse({'conversations': conversations_data})


@csrf_exempt
def fetch_conversation(request, conversation_id):
    print(request.user)
    try:
        try: 
            conversation = Conversation.objects.get(conversation_id=conversation_id)
        except:
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


# @csrf_exempt
def save_news_data():
    try:
        news = News.objects.all().delete()
    except:
        pass
    today = timezone.localtime(timezone.now()).date()
    # List of assets to fetch news data for
    assets_to_fetch = [
        "EURUSD",
        "GBPUSD", 
        "USDJPY",
        "EURGBP" # Add more assets as needed
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
                data=news_entry_data,  # Store the specific fields as JSON data
                day_created=today,  # Use the current datetime as the day_created value
            )
            news_entry.save()
    return JsonResponse({'message': news_data})


@csrf_exempt
def fetch_news_data(request):
    # Check if news data for the current day already exists
    today = timezone.localtime(timezone.now()).date()

    # Check if news data for the current day already exists
    existing_news = News.objects.filter(day_created=today)
    
    if not existing_news.exists():
        # If data for the current day exists, return a message indicating it
        save_news_data()
        # return JsonResponse({'message': 'News data for today already exists.'})
    # else:
        # If data for the current day doesn't exist, fetch and save news data
        # return JsonResponse({'message': f'News data for today does not exist {str(today)}'})
    
    # Fetch all news data without using serializers
    news_objects = News.objects.all()
    
    # Create a list of dictionaries representing the model instances
    news_data = []
    for news in news_objects:
        news_data.append({
            "symbol": news.symbol,
            "description": news.data,
            "created_on": news.day_created,
            'today': today,
        })
    
    # # Convert the list to JSON and return it
    return JsonResponse({"news_data": news_data}, safe=False)
    

@csrf_exempt
async def handle_api_request(type_1, type_2, ma1, ma2, dataframe, backtest_period):

    class SmaCross(Strategy):
        n0 = 18 # Exponential Moving Average
        n1 = 50 # Exponential Moving Average
        n2 = 200 # Simple Moving Average
        equity = 100000
        risk_percentage = 30
        reward_percentage = 100
        # current_price = 0
        reward_ratio = 15
        position_size = 0.01
        current_position = ''
        range = 2
        # 200 SMA
        ma1_type = f'{type_1}_{ma1}'
        # 50 EMA
        ma2_type = f'{type_2}_{ma2}'



        def init(self):
            price = self.data.Close
            self.ma1 = self.I(SMA, price, 50)
            self.ma2 = self.I(SMA, price, 200)
            close = self.data.Close


        def check_moving_averages_for_buy(self, df, range):
            past_10_rows = df[[self.ma2_type, self.ma1_type]].tail(range)
            past_10_rows['Converge'] = past_10_rows[self.ma2_type] < past_10_rows[self.ma1_type]
            past = past_10_rows.tail(1)['Converge'].values[0]
            second_last_row = past_10_rows['Converge'].iloc[-2]
            if past == False and second_last_row == True:
                # print('True')
                return True
            else:
                # print('False')
                return False


        def check_moving_averages_for_sell(self, df, range):
            past_10_rows = df[[self.ma2_type, self.ma1_type]].tail(range)
            past_10_rows['Diverge'] = past_10_rows[self.ma2_type] > past_10_rows[self.ma1_type]
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
            if df.tail(1)[self.ma2_type].values[0] > df.tail(1)[self.ma1_type].values[0]:
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
                    if self.current_position != 'buy':
                        if self.position:
                            self.position.close()
                        self.buy()
                        self.current_position = 'buy'
            elif df.tail(1)[self.ma2_type].values[0] < df.tail(1)[self.ma1_type].values[0]:
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
                    if self.current_position != 'sell':
                        if self.position:
                            self.position.close()
                        self.sell()
                        self.current_position = 'sell'


        def next(self):
            df = pd.DataFrame({'Open': self.data.Open, 'High': self.data.High, 'Low': self.data.Low, 'Close': self.data.Close, 'Volume': self.data.Volume})
            # df['SMA_200'] = df['Close'].rolling(window=200).mean()
            ma1_type = f'{type_1}_{ma1}'
            ma2_type = f'{type_2}_{ma2}'
            if type_1 == 'SMA':
                df[ma1_type] = ta.sma(df["Close"], length=int(ma1))
            elif type_1 == 'EMA':
                df[ma1_type] = ta.ema(df["Close"], length=int(ma1))
            if type_2 == 'SMA':
                df[ma2_type] = ta.sma(df["Close"], length=int(ma2))    
            elif type_2 == 'EMA':
                df[ma2_type] = ta.ema(df["Close"], length=int(ma2))
            
            try:
                self.moving_average(df)
                # print('Running Algorithm...')
                # self.check_moving_averages_for_buy(df, self.range)
            except Exception as e:
                print(f'Error occured: {e}')
                pass
    
    if dataframe == '5Min':
        df_to_use = './XAUUSD5M.csv'
    elif dataframe == '15Min':
        df_to_use = './XAUUSD15M.csv'
    elif dataframe == '30Min':
        df_to_use = './XAUUSD30M.csv'
    elif dataframe == '1H':
        df_to_use = './XAUUSD1H.csv'
    elif dataframe == '4H':
        df_to_use = './XAUUSD4H.csv'
    elif dataframe == '1D':
        df_to_use = './XAUUSD1D.csv'
    
    if backtest_period == '0-25':
        start = 0
        end = 0.25
    elif backtest_period == '25-50':
        start = 0.25
        end = 0.5
    elif backtest_period == '50-75':
        start = 0.5
        end = 0.75
    elif backtest_period == '75-100':
        start = 0.75
        end = 1

        
    df_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), df_to_use)
    df = pd.read_csv(df_path).drop_duplicates()
    df.index = pd.to_datetime(df['Time'].values)
    del df['Time']
    length = int(len(df) * start)
    second_length = int(len(df) * end)
    bt = Backtest(df[length:second_length], SmaCross, exclusive_orders=False, cash=10000)
    output = bt.run()
    
    # Convert the relevant output fields to a dictionary
    result_dict = {
        "Start": str(output['Start']),
        "End": str(output['End']),
        "Duration": str(output['Duration']),
        "Exposure Time [%]": output['Exposure Time [%]'],
        "Equity Final [$]": output['Equity Final [$]'],
        "Equity Peak [$]": output['Equity Peak [$]'],
        "Return [%]": output['Return [%]'],
        "Buy & Hold Return [%]": output['Buy & Hold Return [%]'],
        "Return (Ann.) [%]": output['Return (Ann.) [%]'],
        "Volatility (Ann.) [%]": output['Volatility (Ann.) [%]'],
        "Sharpe Ratio": output['Sharpe Ratio'],
        "Sortino Ratio": output['Sortino Ratio'],
        "Calmar Ratio": output['Calmar Ratio'],
        "Max. Drawdown [%]": output['Max. Drawdown [%]'],
        "Avg. Drawdown [%]": output['Avg. Drawdown [%]'],
        "Max. Drawdown Duration": str(output['Max. Drawdown Duration']),
        "Avg. Drawdown Duration": str(output['Avg. Drawdown Duration']),
        "# Trades": output['# Trades'],
        "Win Rate [%]": output['Win Rate [%]'],
        "Best Trade [%]": output['Best Trade [%]'],
        "Worst Trade [%]": output['Worst Trade [%]'],
        "Avg. Trade [%]": output['Avg. Trade [%]'],
        "Max. Trade Duration": str(output['Max. Trade Duration']),
        "Avg. Trade Duration": str(output['Avg. Trade Duration']),
        "Profit Factor": output['Profit Factor'],
        "Expectancy [%]": output['Expectancy [%]'],
        "SQN": output['SQN'],
    }
    return result_dict


@csrf_exempt
def moving_average_bot(request, type_1, type_2, ma1, ma2, dataframe, backtest_period):

    try:
        MovingAverageBot.objects.all().delete()
        print(f'All deleted successfully!')
    except Exception as e:
        print(f'Exception when deleting is: {e}')
        pass
    new_moving_average_backtest = MovingAverageBot(ma1_type=type_1, ma1=int(ma1), ma2_type=type_2, ma2=int(ma2))
    new_moving_average_backtest.save()

    async def inner():
        result = await handle_api_request(type_1, type_2, ma1, ma2, dataframe, backtest_period)
        return JsonResponse({'Output': result})

    # Run the asynchronous code using the event loop
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(inner())


# https://backend-production-c0ab.up.railway.app/create-bot/sma/ema/200/50


@csrf_exempt
async def handle_api_request_bbands(length, std, dataframe, backtest_period):
    class BBands(Strategy):
        equity = 100000
        risk_percentage = 20
        reward_percentage = 60
        # current_price = 0
        reward_ratio = 15
        position_size = 0.01
        current_position = ''
        upper_band = f'BBU_{length}_{float(std)}'
        middle_band = f'BBM_{length}_{float(std)}'
        bottom_band = f'BBL_{length}_{float(std)}'


        def init(self):
            price = self.data.Close
            self.ma1 = self.I(SMA, price, 10)
            self.ma2 = self.I(SMA, price, 20)


        def bbands(self, df):
            if df.tail(1)['Close'].values[0] >= df.tail(1)[self.upper_band].values[0]:
                price = self.data.Close[-1]
                gain_amount = self.reward_percentage
                risk_amount = self.risk_percentage 
                tp_level = price + gain_amount
                sl_level = price - risk_amount

                # self.buy(sl=sl_level)
                if self.current_position != 'buy':
                    if self.position:
                        self.position.close()
                    self.buy()
                    self.current_position = 'buy'
            elif df.tail(1)['Close'].values[0] <= df.tail(1)[self.bottom_band].values[0]:
                price = self.data.Close[-1]
                gain_amount = self.reward_percentage
                risk_amount = self.risk_percentage 
                tp_level = price - gain_amount
                sl_level = price + risk_amount

                # self.sell(sl=sl_level)
                if self.current_position != 'sell':
                    if self.position:
                        self.position.close()
                    self.sell()
                    self.current_position = 'sell'


        def next(self):
            df = pd.DataFrame({'Open': self.data.Open, 'High': self.data.High, 'Low': self.data.Low, 'Close': self.data.Close, self.upper_band: self.data[self.upper_band], self.bottom_band: self.data[self.bottom_band]})
            # current_close = df['Close']
            # print('1')

            # print(f'first is {self.data.upper_band}')
            current_close = ta.bbands(close=df['Close'], length=int(length), std=int(std), append=True)
            # print(f'current_close is {current_close}')
            # print('2')
            try:
                # print('3')
                # print(f'Running Algorithm...')
                self.bbands(df)
            except Exception as e:
                print(f'Exception is {e}')
                pass

    if dataframe == '5Min':
        df_to_use = './XAUUSD5M.csv'
    elif dataframe == '15Min':
        df_to_use = './XAUUSD15M.csv'
    elif dataframe == '30Min':
        df_to_use = './XAUUSD30M.csv'
    elif dataframe == '1H':
        df_to_use = './XAUUSD1H.csv'
    elif dataframe == '4H':
        df_to_use = './XAUUSD4H.csv'
    elif dataframe == '1D':
        df_to_use = './XAUUSD1D.csv'
    
    if backtest_period == '0-25':
        start = 0
        end = 0.25
    elif backtest_period == '25-50':
        start = 0.25
        end = 0.5
    elif backtest_period == '50-75':
        start = 0.5
        end = 0.75
    elif backtest_period == '75-100':
        start = 0.75
        end = 1

    # comment
    df_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), df_to_use)
    df = pd.read_csv(df_path).drop_duplicates()
    df.index = pd.to_datetime(df['Time'].values)
    del df['Time']
    current_close = ta.bbands(close=df['Close'], length=int(length), std=int(std), append=True)
    upper_band = f'BBU_{length}_{float(std)}'
    middle_band = f'BBM_{length}_{float(std)}'
    bottom_band = f'BBL_{length}_{float(std)}'

    print(f'current close first is {current_close}')
    df[upper_band] = current_close[upper_band]
    df[middle_band] = current_close[middle_band]
    df[bottom_band] = current_close[bottom_band]
    print(f'first df is {df}')
    length = int(len(df) * start)
    second_length = int(len(df) * end)
    bt = Backtest(df[length:second_length], BBands, exclusive_orders=False, cash=10000)
    output = bt.run()
    
    # Convert the relevant output fields to a dictionary
    result_dict = {
        "Start": str(output['Start']),
        "End": str(output['End']),
        "Duration": str(output['Duration']),
        "Exposure Time [%]": output['Exposure Time [%]'],
        "Equity Final [$]": output['Equity Final [$]'],
        "Equity Peak [$]": output['Equity Peak [$]'],
        "Return [%]": output['Return [%]'],
        "Buy & Hold Return [%]": output['Buy & Hold Return [%]'],
        "Return (Ann.) [%]": output['Return (Ann.) [%]'],
        "Volatility (Ann.) [%]": output['Volatility (Ann.) [%]'],
        "Sharpe Ratio": output['Sharpe Ratio'],
        "Sortino Ratio": output['Sortino Ratio'],
        "Calmar Ratio": output['Calmar Ratio'],
        "Max. Drawdown [%]": output['Max. Drawdown [%]'],
        "Avg. Drawdown [%]": output['Avg. Drawdown [%]'],
        "Max. Drawdown Duration": str(output['Max. Drawdown Duration']),
        "Avg. Drawdown Duration": str(output['Avg. Drawdown Duration']),
        "# Trades": output['# Trades'],
        "Win Rate [%]": output['Win Rate [%]'],
        "Best Trade [%]": output['Best Trade [%]'],
        "Worst Trade [%]": output['Worst Trade [%]'],
        "Avg. Trade [%]": output['Avg. Trade [%]'],
        "Max. Trade Duration": str(output['Max. Trade Duration']),
        "Avg. Trade Duration": str(output['Avg. Trade Duration']),
        "Profit Factor": output['Profit Factor'],
        "Expectancy [%]": output['Expectancy [%]'],
        "SQN": output['SQN'],
    }
    return result_dict


@csrf_exempt
def bbands_bot(request, length, std, dataframe, backtest_period):
    async def inner_bband():
        result = await handle_api_request_bbands(length, std, dataframe, backtest_period)
        return JsonResponse({'Output': result})

    # Run the asynchronous code using the event loop
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(inner_bband())


@csrf_exempt
async def handle_api_request_rsi(length, overbought_level, oversold_level, dataframe, backtest_period):
    print('1 is here!!!!!!!!!!!')
    class RSI(Strategy):
        equity = 100000
        risk_percentage = 20
        reward_percentage = 50
        # current_price = 0
        reward_ratio = 15
        position_size = 0.01
        current_position = ''

        def init(self):
            price = self.data.Close
            self.ma1 = self.I(SMA, price, 10)
            self.ma2 = self.I(SMA, price, 20)


        def rsi(self, df):
            if df.tail(1)['RSI'].values[0] > int(overbought_level):
                price = self.data.Close[-1]
                gain_amount = self.reward_percentage
                risk_amount = self.risk_percentage
                tp_level = price - gain_amount
                sl_level = price + risk_amount
                # self.sell(tp=tp_level, sl=sl_level)
                if self.current_position != 'sell':
                    if self.position:
                        self.position.close()
                    self.sell(sl=sl_level)
                    self.current_position = 'sell'
            elif df.tail(1)['RSI'].values[0] < int(oversold_level):
                price = self.data.Close[-1]
                gain_amount = self.reward_percentage
                risk_amount = self.risk_percentage
                tp_level = price + gain_amount
                sl_level = price - risk_amount
                # self.buy(tp=tp_level,sl=sl_level)
                if self.current_position != 'buy':
                    if self.position:
                        self.position.close()
                    self.buy(sl=sl_level)
                    self.current_position = 'buy'


        def next(self):
            print(f'self.data is {self.data}')
            df = pd.DataFrame({'Open': self.data.Open, 'High': self.data.High, 'Low': self.data.Low, 'Close': self.data.Close, 'RSI': self.data.RSI})
            print(f'df is {df}')
            try:
                self.rsi(df)
            except Exception as e:
                print(f'df is {df}')
                print(f'Exception is {e}')
                pass        

            
    if dataframe == '5Min':
        df_to_use = './XAUUSD5M.csv'
    elif dataframe == '15Min':
        df_to_use = './XAUUSD15M.csv'
    elif dataframe == '30Min':
        df_to_use = './XAUUSD30M.csv'
    elif dataframe == '1H':
        df_to_use = './XAUUSD1H.csv'
    elif dataframe == '4H':
        df_to_use = './XAUUSD4H.csv'
    elif dataframe == '1D':
        df_to_use = './XAUUSD1D.csv'
    
    if backtest_period == '0-25':
        start = 0
        end = 0.25
    elif backtest_period == '25-50':
        start = 0.25
        end = 0.5
    elif backtest_period == '50-75':
        start = 0.5
        end = 0.75
    elif backtest_period == '75-100':
        start = 0.75
        end = 1
                
    df_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), df_to_use)
    df = pd.read_csv(df_path).drop_duplicates()
    df.index = pd.to_datetime(df['Time'].values)
    print(f'No error here')
    del df['Time']
    df['RSI'] = ta.rsi(df['Close'], length = int(length))
    print(f'df 1 is {df}')
    length = int(len(df) * start)
    second_length = int(len(df) * end)
    bt = Backtest(df[length:second_length], RSI, exclusive_orders=False, cash=10000)
    output = bt.run()
    
    # Convert the relevant output fields to a dictionary
    result_dict = {
        "Start": str(output['Start']),
        "End": str(output['End']),
        "Duration": str(output['Duration']),
        "Exposure Time [%]": output['Exposure Time [%]'],
        "Equity Final [$]": output['Equity Final [$]'],
        "Equity Peak [$]": output['Equity Peak [$]'],
        "Return [%]": output['Return [%]'],
        "Buy & Hold Return [%]": output['Buy & Hold Return [%]'],
        "Return (Ann.) [%]": output['Return (Ann.) [%]'],
        "Volatility (Ann.) [%]": output['Volatility (Ann.) [%]'],
        "Sharpe Ratio": output['Sharpe Ratio'],
        "Sortino Ratio": output['Sortino Ratio'],
        "Calmar Ratio": output['Calmar Ratio'],
        "Max. Drawdown [%]": output['Max. Drawdown [%]'],
        "Avg. Drawdown [%]": output['Avg. Drawdown [%]'],
        "Max. Drawdown Duration": str(output['Max. Drawdown Duration']),
        "Avg. Drawdown Duration": str(output['Avg. Drawdown Duration']),
        "# Trades": output['# Trades'],
        "Win Rate [%]": output['Win Rate [%]'],
        "Best Trade [%]": output['Best Trade [%]'],
        "Worst Trade [%]": output['Worst Trade [%]'],
        "Avg. Trade [%]": output['Avg. Trade [%]'],
        "Max. Trade Duration": str(output['Max. Trade Duration']),
        "Avg. Trade Duration": str(output['Avg. Trade Duration']),
        "Profit Factor": output['Profit Factor'],
        "Expectancy [%]": output['Expectancy [%]'],
        "SQN": output['SQN'],
    }
    return result_dict


@csrf_exempt
def rsi_bot(request, length, overbought_level, oversold_level, dataframe, backtest_period):
    # oversold_level = int(oversold_level.remove(f'{length}_'))
    print('Hello World!!!!!!!!!!!!!!!!!')
    async def inner_rsi():
        print('Code runnning...')
        # print(f'Length is {length}. Overbought Level is {overbought_level}. Oversold Level is {oversold_level}.')
        result = await handle_api_request_rsi(length, overbought_level, oversold_level, dataframe, backtest_period)
        return JsonResponse({'Output': result})

    # Run the asynchronous code using the event loop
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(inner_rsi())


@csrf_exempt
async def handle_api_request_momentum(dataframe, backtest_period):

    class Momentum(Strategy):
        equity = 100000
        risk_percentage = 20
        reward_percentage = 50
        # current_price = 0
        reward_ratio = 15
        position_size = 0.01
        current_position = ''

        def init(self):
            price = self.data.Close
            self.ma1 = self.I(SMA, price, 10)
            self.ma2 = self.I(SMA, price, 20)


        def momentum(self, df):

            if df.tail(1)['MOM'].values[0] > 80:
                price = self.data.Close[-1]
                gain_amount = self.reward_percentage
                risk_amount = self.risk_percentage
                tp_level = price - gain_amount
                sl_level = price + risk_amount

                # if self.current_position != 'sell':
                if self.position:
                    self.position.close()
                self.sell()
                    # self.current_position = 'sell'
                    # self.sell()

            elif df.tail(1)['MOM'].values[0] < 20:
                price = self.data.Close[-1]
                gain_amount = self.reward_percentage
                risk_amount = self.risk_percentage
                tp_level = price + gain_amount
                sl_level = price - risk_amount
                
                # if self.current_position != 'buy':
                if self.position:
                    self.position.close()
                self.buy()
                # self.current_position = 'buy'
                    # self.buy()


        def next(self):
            df = pd.DataFrame({'Open': self.data.Open, 'High': self.data.High, 'Low': self.data.Low, 'Close': self.data.Close, 'Volume': self.data.Volume})
            df['MOM'] = ta.mom(df['Close'])
            # if not self.position:
            try:
                self.momentum(df)
                # print('Running Backtesting Algorithm...')
            except Exception as e:
                print(f'Exception is {e}')
                pass
    

    if dataframe == '5Min':
        df_to_use = './XAUUSD5M.csv'
    elif dataframe == '15Min':
        df_to_use = './XAUUSD15M.csv'
    elif dataframe == '30Min':
        df_to_use = './XAUUSD30M.csv'
    elif dataframe == '1H':
        df_to_use = './XAUUSD1H.csv'
    elif dataframe == '4H':
        df_to_use = './XAUUSD4H.csv'
    elif dataframe == '1D':
        df_to_use = './XAUUSD1D.csv'
    

    if backtest_period == '0-25':
        start = 0
        end = 0.25
    elif backtest_period == '25-50':
        start = 0.25
        end = 0.5
    elif backtest_period == '50-75':
        start = 0.5
        end = 0.75
    elif backtest_period == '75-100':
        start = 0.75
        end = 1


    df_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), df_to_use)
    df = pd.read_csv(df_path).drop_duplicates()
    df.index = pd.to_datetime(df['Time'].values)
    del df['Time']
    # test_length = int(len(df) * 0.25)
    length = int(len(df) * start)
    second_length = int(len(df) * end)
    bt = Backtest(df[length:second_length], Momentum, exclusive_orders=False, cash=10000)
    output = bt.run()
    
    # Convert the relevant output fields to a dictionary
    result_dict = {
        "Start": str(output['Start']),
        "End": str(output['End']),
        "Duration": str(output['Duration']),
        "Exposure Time [%]": output['Exposure Time [%]'],
        "Equity Final [$]": output['Equity Final [$]'],
        "Equity Peak [$]": output['Equity Peak [$]'],
        "Return [%]": output['Return [%]'],
        "Buy & Hold Return [%]": output['Buy & Hold Return [%]'],
        "Return (Ann.) [%]": output['Return (Ann.) [%]'],
        "Volatility (Ann.) [%]": output['Volatility (Ann.) [%]'],
        "Sharpe Ratio": output['Sharpe Ratio'],
        "Sortino Ratio": output['Sortino Ratio'],
        "Calmar Ratio": output['Calmar Ratio'],
        "Max. Drawdown [%]": output['Max. Drawdown [%]'],
        "Avg. Drawdown [%]": output['Avg. Drawdown [%]'],
        "Max. Drawdown Duration": str(output['Max. Drawdown Duration']),
        "Avg. Drawdown Duration": str(output['Avg. Drawdown Duration']),
        "# Trades": output['# Trades'],
        "Win Rate [%]": output['Win Rate [%]'],
        "Best Trade [%]": output['Best Trade [%]'],
        "Worst Trade [%]": output['Worst Trade [%]'],
        "Avg. Trade [%]": output['Avg. Trade [%]'],
        "Max. Trade Duration": str(output['Max. Trade Duration']),
        "Avg. Trade Duration": str(output['Avg. Trade Duration']),
        "Profit Factor": output['Profit Factor'],
        "Expectancy [%]": output['Expectancy [%]'],
        "SQN": output['SQN'],
    }
    return result_dict


@csrf_exempt
def momentum_bot(request, dataframe, backtest_period):
    async def inner_momentum():
        result = await handle_api_request_momentum(dataframe, backtest_period)
        return JsonResponse({'Output': result})

    # Run the asynchronous code using the event loop
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(inner_momentum())


@csrf_exempt
async def handle_api_request_candlesticks(engulfing, pinbar, morning_star, three_white_soldiers, doji_star, methods, dataframe, backtest_period):
    class Strat(Strategy):

        current_day = 0
        equity = 100000
        risk_percentage = 20
        reward_percentage = 50
        # current_price = 0
        reward_ratio = 15
        position_size = 0.01
        # path = '/content/trading-bot/candlestick_chart.png'
        # sr_path  = '/content/sar'
        candlestick_backtrack = 288


        def init(self):
            # super().init()
            # super().set_trailing_sl(3)
            close = self.data.Close
    

        def bullish_engulfing(self, df):

            df_test = df.tail(6)
            df_test = df_test.drop_duplicates()
            test_size = len(df)
            num_engulfing = 0

            for i in range(test_size-1):
                first_candle = df_test.iloc[i-1]
                second_candle = df_test.iloc[i-2]
                third_candle  = df_test.iloc[i-3]
                fourth_candle = df_test.iloc[i-4]
                fifth_candle = df_test.iloc[i-5]
                second_test = first_candle.Close > second_candle.Open

                if is_bearish_candle(second_candle) and is_bullish_candle(first_candle) and second_test == True and is_bearish_run(fifth_candle, fourth_candle, third_candle, second_candle):
                    # num_engulfing += 1
                    # print('Bullish Engulfing')
                    if df.tail(1)['EMA_50'].values[0] > df.tail(1)['SMA_200'].values[0]:
                        # # Set the style of the plot
                        # df.index = pd.to_datetime(df.index)
                        # style = mpf.make_mpf_style(base_mpf_style='classic')
                        # Create the figure object without plotting
                        # fig, axes = mpf.plot(df.tail(self.candlestick_backtrack), type='candle', volume=True, returnfig=True, style=style)
                        # plt.close(fig)
                        # # Save the figure to a file
                        # fig.savefig('candlestick_chart.png')
                        # # if self.position:
                        # #     self.position.close()

                        # if process_image(self.path) == 2:
                        price = self.data.Close[-1]
                        gain_amount = self.reward_percentage
                        risk_amount = self.risk_percentage
                        tp_level = price + self.reward_percentage
                        sl_level = price - self.risk_percentage

                        # levels = get_fibonacci_levels(df=df.tail(75), trend='uptrend')
                        # thirty_eight_retracement = levels[2]
                        # sixty_one8_retracement = levels[4]
                        # if thirty_eight_retracement <= price <= sixty_one8_retracement:
                            # self.position.close()
                        if self.position:
                            self.position.close()
                        self.buy(tp=tp_level, sl=sl_level)
                break


        def bearish_engulfing(self, df):
            df_test = df.tail(6)
            df_test = df_test.drop_duplicates()
            test_size = len(df)
            num_engulfing = 0

            for i in range(test_size-1):
                first_candle = df_test.iloc[i-1]
                second_candle = df_test.iloc[i-2]
                third_candle  = df_test.iloc[i-3]
                fourth_candle = df_test.iloc[i-4]
                fifth_candle = df_test.iloc[i-5]
                # first_test = first_candle.Open < second_candle.Close
                second_test = first_candle.Close < second_candle.Open

                if is_bullish_candle(second_candle) and is_bearish_candle(first_candle) and second_test == True and is_bullish_run(fifth_candle, fourth_candle, third_candle, second_candle):
                    # num_engulfing += 1
                    # print('Bearish Engulfing')
                    price = self.data.Close[-1]

                    if df.tail(1)['EMA_50'].values[0] < df.tail(1)['SMA_200'].values[0]:
                        # df.index = pd.to_datetime(df.index)
                        # style = mpf.make_mpf_style(base_mpf_style='classic')

                        # # Create the figure object without plotting
                        # fig, axes = mpf.plot(df.tail(self.candlestick_backtrack), type='candle', volume=True, returnfig=True, style=style)
                        # plt.close(fig)
                        # # Save the figure to a file
                        # fig.savefig('candlestick_chart.png')

                        # if self.position:
                        #     self.position.close()
                            # pass
                        # if process_image(self.path) == 0:
                        gain_amount = self.reward_percentage
                        risk_amount = self.risk_percentage
                        tp_level = price - self.reward_percentage
                        sl_level = price + self.risk_percentage
                        # levels = get_fibonacci_levels(df=df.tail(75), trend='downtrend')
                        # thirty_eight_retracement = levels[2]
                        # sixty_one8_retracement = levels[4]
                            # if thirty_eight_retracement <= price <= sixty_one8_retracement:
                            # self.position.close()
                        if self.position:
                            self.position.close()
                        self.sell(tp=tp_level, sl=sl_level)
                break

            
        
        def bullish_pinbar(self, df):
            dataframe = df.drop_duplicates()
            df = df.tail(1)
            df = df.drop_duplicates()
            test_size = len(df)
            num_pin_bars = 0
            price = self.data.Close[-1]

            for i in range(test_size-1):
                candle = df.iloc[i]
                is_pin_bar = (candle.Close - candle.Low) > 5
                if is_pin_bar:
                    num_pin_bars += 1
                    # print('Bullish Pin Bar')
                    price = self.data.Close[-1]
                    gain_amount = self.reward_percentage
                    risk_amount = self.risk_percentage
                    tp_level = price + gain_amount
                    sl_level = price - risk_amount

                    # dataframe.index = pd.to_datetime(dataframe.index)
                    # style = mpf.make_mpf_style(base_mpf_style='classic')

                    # Create the figure object without plotting
                    # fig, axes = mpf.plot(dataframe.tail(self.candlestick_backtrack), type='candle', volume=True, returnfig=True, style=style)
                    # plt.close(fig)
                    # Save the figure to a file
                    # fig.savefig('candlestick_chart.png')

                    if df.tail(1)['EMA_50'].values[0] > df.tail(1)['SMA_200'].values[0]:
                        self.buy(tp=tp_level, sl=sl_level)


        def bearish_pinbar(self, df):
            dataframe = df.drop_duplicates()
            df = df.tail(1)
            df = df.drop_duplicates()
            test_size = len(df)
            num_pin_bars = 0

            for i in range(test_size-1):
                candle = df.iloc[i]
                is_pin_bar = abs(candle.Close - candle.High) <  5
                if is_pin_bar:
                    num_pin_bars += 1
                    # print('Bearish Pin Bar')
                    price = self.data.Close[-1]
                    gain_amount = self.reward_percentage
                    risk_amount = self.risk_percentage
                    tp_level = price - gain_amount
                    sl_level = price + risk_amount

                    # dataframe.index = pd.to_datetime(dataframe.index)
                    # style = mpf.make_mpf_style(base_mpf_style='classic')

                    # Create the figure object without plotting
                    # fig, axes = mpf.plot(dataframe.tail(self.candlestick_backtrack), type='candle', volume=True, returnfig=True, style=style)
                    # plt.close(fig)
                    # Save the figure to a file
                    # fig.savefig('candlestick_chart.png')

                    if df.tail(1)['EMA_50'].values[0] < df.tail(1)['SMA_200'].values[0]:
                        self.sell(tp=tp_level, sl=sl_level)


        def doji_star(self, df):
            # print('')
            df = df.drop_duplicates()
            dataframe = df
            df = df.tail(5)
            test_size = len(df)
            bullish_doji = 0
            bearish_doji = 0

            for i in range(test_size-4):
                first_prev_candle = df.iloc[i]
                second_prev_candle = df.iloc[i+1]
                third_prev_candle = df.iloc[i+2]
                prev_candle = df.iloc[i+3]
                testing_candle = df.iloc[i+4]
                price = self.data.Close[-1]

                if is_bullish_run(first_prev_candle, second_prev_candle, third_prev_candle, prev_candle):
                    test1 = testing_candle.High - testing_candle.Close
                    test2 = testing_candle.Close - testing_candle.Low
                    if test1 == test2:
                        bullish_doji += 1
                        # print('bullish doji star')
                        price = self.data.Close[-1]
                        gain_amount = self.reward_percentage
                        risk_amount = self.risk_percentage
                        tp_level = price - gain_amount
                        sl_level = price + risk_amount

                        # dataframe.index = pd.to_datetime(dataframe.index)
                        # style = mpf.make_mpf_style(base_mpf_style='classic')

                        # Create the figure object without plotting
                        # fig, axes = mpf.plot(dataframe.tail(self.candlestick_backtrack), type='candle', volume=True, returnfig=True, style=style)
                        # plt.close(fig)
                        # Save the figure to a file
                        # fig.savefig('candlestick_chart.png')

                        if df.tail(1)['EMA_50'].values[0] < df.tail(1)['SMA_200'].values[0]:
                            self.sell(tp=tp_level, sl=sl_level)
                elif is_bearish_run(first_prev_candle, second_prev_candle, third_prev_candle, prev_candle):
                    test1 = testing_candle.Open - testing_candle.Close
                    test2 = testing_candle.Close - testing_candle.Low
                    if test1 == test2:
                        bearish_doji += 1
                        price = self.data.Close[-1]
                        gain_amount = self.reward_percentage
                        risk_amount = self.risk_percentage
                        tp_level = price + gain_amount
                        sl_level = price - risk_amount

                        # dataframe.index = pd.to_datetime(dataframe.index)
                        # style = mpf.make_mpf_style(base_mpf_style='classic')

                        # Create the figure object without plotting
                        # fig, axes = mpf.plot(dataframe.tail(self.candlestick_backtrack), type='candle', volume=True, returnfig=True, style=style)
                        # plt.close(fig)
                        # Save the figure to a file
                        # fig.savefig('candlestick_chart.png')

                        if df.tail(1)['EMA_50'].values[0] > df.tail(1)['SMA_200'].values[0]:
                            self.buy(tp=tp_level, sl=sl_level)

        def three_white_soldier(self, df):
            # print('')
            dataframe = df.drop_duplicates()
            df = df.drop_duplicates()
            df = df.tail(6)
            test_size = len(df)
            three_white_soldiers = 0
            three_black_crows = 0

            for i in range(test_size-5):
                first_prev_candle = df.iloc[i]
                second_prev_candle = df.iloc[i+1]
                third_prev_candle = df.iloc[i+2]
                prev_candle = df.iloc[i+3]
                testing_candle = df.iloc[i+4]
                testing_candle_2 = df.iloc[i+5]
                price = self.data.Close[-1]

                if is_bearish_run(first_prev_candle, second_prev_candle, third_prev_candle, prev_candle):
                    if testing_candle_2.Close > testing_candle.Close and testing_candle.Close > prev_candle.Close:
                        three_white_soldiers += 1
                        # print('bullish three white soldiers')
                        if df.tail(1)['EMA_50'].values[0] > df.tail(1)['SMA_200'].values[0]:
                            # dataframe.index = pd.to_datetime(dataframe.index)
                            # style = mpf.make_mpf_style(base_mpf_style='classic')

                            # Create the figure object without plotting
                            # fig, axes = mpf.plot(dataframe.tail(self.candlestick_backtrack), type='candle', volume=True, returnfig=True, style=style)
                            # plt.close(fig)
                            # Save the figure to a file
                            # fig.savefig('candlestick_chart.png')
                            # plt.close(fig)
                            # if self.position:
                            #   self.position.close()
                            # if process_image(self.path) == 2:
                            price = self.data.Close[-1]
                            gain_amount = self.reward_percentage
                            risk_amount = self.risk_percentage
                            tp_level = price + self.reward_percentage
                            sl_level = price - self.risk_percentage
                            # levels = get_fibonacci_levels(df=dataframe.tail(75), trend='uptrend')
                            # thirty_eight_retracement = levels[2]
                            # sixty_one8_retracement = levels[4]
                                # if thirty_eight_retracement <= prev_candle.Close <= sixty_one8_retracement:
                                # self.position.close()
                            self.buy(tp=tp_level, sl=sl_level)
                elif is_bullish_run(first_prev_candle, second_prev_candle, third_prev_candle, prev_candle):
                    if testing_candle_2.Close < testing_candle.Close and testing_candle.Close < prev_candle.Close:
                        three_black_crows += 1
                        # print('bearish three black crows')
                        if df.tail(1)['EMA_50'].values[0] < df.tail(1)['SMA_200'].values[0]:
                            # df.index = pd.to_datetime(df.index)
                            # style = mpf.make_mpf_style(base_mpf_style='classic')

                            # Create the figure object without plotting
                            # fig, axes = mpf.plot(df.tail(self.candlestick_backtrack), type='candle', volume=True, returnfig=True, style=style)
                            # plt.close(fig)
                            # Save the figure to a file
                            # fig.savefig('candlestick_chart.png')
                            # plt.close(fig)
                            # if self.position:
                            #   self.position.close()
                            # if process_image(self.path) == 0:
                            price = self.data.Close[-1]
                            gain_amount = self.reward_percentage 
                            risk_amount = self.risk_percentage
                            tp_level = price - self.reward_percentage
                            sl_level = price + self.risk_percentage
                            # levels = get_fibonacci_levels(df=dataframe.tail(75), trend='downtrend')
                            # thirty_eight_retracement = levels[2]
                            # sixty_one8_retracement = levels[4]
                                # if thirty_eight_retracement <= prev_candle.Close <= sixty_one8_retracement:
                                # self.position.close()
                            self.sell(tp=tp_level, sl=sl_level)


        def morning_star(self, df):
            # print('')
            dataframe = df.drop_duplicates()
            df = df.drop_duplicates()
            df = df.tail(6)
            test_size = len(df)
            morning_stars = 0
            evening_stars = 0
            price = self.data.Close[-1]

            for i in range(test_size-5):
                first_prev_candle = df.iloc[i]
                second_prev_candle = df.iloc[i+1]
                third_prev_candle = df.iloc[i+2]
                prev_candle = df.iloc[i+3]
                testing_candle = df.iloc[i+4]
                testing_candle_2 = df.iloc[i+5]

                if is_bearish_run(first_prev_candle, second_prev_candle, third_prev_candle, prev_candle):
                    test = testing_candle.Open - testing_candle.Close
                    if testing_candle_2.Close > testing_candle.Close and 0 < test < 2:
                        morning_stars += 1
                        # print('bullish morning star')
                        price = self.data.Close[-1]
                        gain_amount = self.reward_percentage
                        risk_amount = self.risk_percentage
                        tp_level = price + gain_amount
                        sl_level = price - risk_amount

                        # dataframe.index = pd.to_datetime(dataframe.index)
                        # style = mpf.make_mpf_style(base_mpf_style='classic')

                        # Create the figure object without plotting
                        # fig, axes = mpf.plot(dataframe.tail(self.candlestick_backtrack), type='candle', volume=True, returnfig=True, style=style)

                        # Save the figure to a file
                        # fig.savefig('candlestick_chart.png')
                        # plt.close(fig)
                        if df.tail(1)['EMA_50'].values[0] > df.tail(1)['SMA_200'].values[0]:
                            # levels = get_fibonacci_levels(df=dataframe.tail(75), trend='uptrend')
                            # thirty_eight_retracement = levels[2]
                            # sixty_one8_retracement = levels[4]
                            # if thirty_eight_retracement <= testing_candle.Close <= sixty_one8_retracement:
                            self.buy(tp=tp_level, sl=sl_level)

                elif is_bullish_run(first_prev_candle, second_prev_candle, third_prev_candle, prev_candle):
                    test = testing_candle.Open - testing_candle.Close
                    if testing_candle_2.Close < testing_candle.Close and 0 < test < 2 and testing_candle.Close < prev_candle.Close:
                        evening_stars += 1
                        # print('bearish morning star')
                        price = self.data.Close[-1]
                        gain_amount = self.reward_percentage
                        risk_amount = self.risk_percentage
                        tp_level = price - gain_amount
                        sl_level = price + risk_amount

                        # dataframe.index = pd.to_datetime(dataframe.index)
                        # style = mpf.make_mpf_style(base_mpf_style='classic')

                        # Create the figure object without plotting
                        # fig, axes = mpf.plot(dataframe.tail(self.candlestick_backtrack), type='candle', volume=True, returnfig=True, style=style)

                        # Save the figure to a file
                        # fig.savefig('candlestick_chart.png')
                        # plt.close(fig)
                        if df.tail(1)['EMA_50'].values[0] < df.tail(1)['SMA_200'].values[0]:
                            # levels = get_fibonacci_levels(df=dataframe.tail(75), trend='downtrend')
                            # thirty_eight_retracement = levels[2]
                            # sixty_one8_retracement = levels[4]
                            # if thirty_eight_retracement <= testing_candle.Close <= sixty_one8_retracement:
                            self.sell(tp=tp_level, sl=sl_level)

            

        def methods(self, df):
            dataframe = df.drop_duplicates()
            df = df.drop_duplicates()
            df = df.tail(8)
            test_size = len(df)
            rising_methods = 0
            falling_methods = 0
            price = self.data.Close[-1]

            for i in range(test_size-7):
                first_prev_candle = df.iloc[i]
                second_prev_candle = df.iloc[i+1]
                third_prev_candle = df.iloc[i+2]
                prev_candle = df.iloc[i+3]
                testing_candle = df.iloc[i+4]
                testing_candle_2 = df.iloc[i+5]
                testing_candle_3 = df.iloc[i+6]
                final_candle = df.iloc[7]

                if is_bullish_run(first_prev_candle, second_prev_candle, third_prev_candle, prev_candle) and testing_candle.Close < prev_candle.Close and is_bearish_run_3(testing_candle, testing_candle_2, testing_candle_3):
                    if final_candle.Close > prev_candle.Close:
                        rising_methods += 1
                        # print('rising three methods')
                        price = self.data.Close[-1]
                        gain_amount = self.reward_percentage
                        risk_amount = self.risk_percentage
                        tp_level = price + self.reward_percentage
                        sl_level = price - self.risk_percentage

                        # dataframe.index = pd.to_datetime(dataframe.index)
                        # style = mpf.make_mpf_style(base_mpf_style='classic')

                        # Create the figure object without plotting
                        # fig, axes = mpf.plot(dataframe.tail(self.candlestick_backtrack), type='candle', volume=True, returnfig=True, style=style)
                        # plt.close(fig)
                        # Save the figure to a file
                        # fig.savefig('candlestick_chart.png')
                        # plt.close(fig)

                        if df.tail(1)['EMA_50'].values[0] > df.tail(1)['SMA_200'].values[0]:
                            # levels = get_fibonacci_levels(df=dataframe.tail(75), trend='uptrend')
                            # thirty_eight_retracement = levels[2]
                            # sixty_one8_retracement = levels[4]
                            # if thirty_eight_retracement <= testing_candle_3.Close <= sixty_one8_retracement:
                            self.buy(tp=tp_level, sl=sl_level)

                elif is_bearish_run(first_prev_candle, second_prev_candle, third_prev_candle, prev_candle) and testing_candle.Close > prev_candle.Close and is_bullish_run_3(testing_candle, testing_candle_2, testing_candle_3):
                    if final_candle.Close < prev_candle.Close:
                        falling_methods += 1
                        # print('falling three methods')
                        price = self.data.Close[-1]
                        gain_amount = self.reward_percentage * self.equity
                        risk_amount = self.risk_percentage * self.equity
                        tp_level = price - self.reward_percentage
                        sl_level = price + self.risk_percentage

                        # dataframe.index = pd.to_datetime(dataframe.index)
                        # style = mpf.make_mpf_style(base_mpf_style='classic')

                        # Create the figure object without plotting
                        # fig, axes = mpf.plot(dataframe.tail(self.candlestick_backtrack), type='candle', volume=True, returnfig=True, style=style)
                        # plt.close(fig)
                        # Save the figure to a file
                        # fig.savefig('candlestick_chart.png')
                        # plt.close(fig)

                        if df.tail(1)['EMA_50'].values[0] < df.tail(1)['SMA_200'].values[0]:
                            # levels = get_fibonacci_levels(df=dataframe.tail(75), trend='downtrend')
                            # thirty_eight_retracement = levels[2]
                            # sixty_one8_retracement = levels[4]
                            # if thirty_eight_retracement <= testing_candle_3.Close <= sixty_one8_retracement:
                            self.sell(tp=tp_level, sl=sl_level)
        
        def analyze_candlesticks(self, df):
            if not self.position and engulfing:
                self.bullish_engulfing(df=df)
                self.bearish_engulfing(df=df)
            if not self.position and three_white_soldiers:
                self.three_white_soldier(df=df)
            if not self.position and methods:
                self.methods(df=df)
            if not self.position and morning_star:
                self.morning_star(df=df)
        # self.support_and_resistance(df=df)
            if not self.position and pinbar:
                self.bullish_pinbar(df=df)
                # if not self.position:
                self.bearish_pinbar(df=df)
        # if not self.position:
        #   self.shooting_star(df=df)
            if not self.position and doji_star:
                self.doji_star(df=df)
        # if not self.position:
        # if not self.position:
        #   self.matching(df=df)


        def next(self):
            # super().next()
            # Creating a Pandas DataFrame
            # print(self.data)
            df = pd.DataFrame({'Open': self.data.Open, 'High': self.data.High, 'Low': self.data.Low, 'Close': self.data.Close, 'SMA_200': self.data.SMA_200, 'EMA_50': self.data.EMA_50})
            # df.dropna(inplace=True)
            df = df.fillna(0)
            new_day = self.data.index[-1].day
            mod = new_day % 5
            # print(df)
            # if mod == 0 and self.position:
            #   self.position.close()

            # if self.current_day < new_day and self.position:
            #     self.position.close()
            # self.current_day = new_day
            if not self.position:
                try:
                    self.analyze_candlesticks(df=df)
                except Exception as e:
                    print(f'Error occured here: {e}')
                    pass

    
    if dataframe == '5Min':
        df_to_use = './XAUUSD5M.csv'
    elif dataframe == '15Min':
        df_to_use = './XAUUSD15M.csv'
    elif dataframe == '30Min':
        df_to_use = './XAUUSD30M.csv'
    elif dataframe == '1H':
        df_to_use = './XAUUSD1H.csv'
    elif dataframe == '4H':
        df_to_use = './XAUUSD4H.csv'
    elif dataframe == '1D':
        df_to_use = './XAUUSD1D.csv'
    

    if backtest_period == '0-25':
        start = 0
        end = 0.25
    elif backtest_period == '25-50':
        start = 0.25
        end = 0.5
    elif backtest_period == '50-75':
        start = 0.5
        end = 0.75
    elif backtest_period == '75-100':
        start = 0.75
        end = 1

    
                
    df_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), df_to_use)
    df = pd.read_csv(df_path).drop_duplicates()
    df.index = pd.to_datetime(df['Time'].values)
    del df['Time']
    df["SMA_200"] = ta.sma(df["Close"], length=200)
    df["EMA_50"] = ta.ema(df["Close"], length=50)
    length = int(len(df) * start)
    second_length = int(len(df) * end)
    bt = Backtest(df[length:second_length], Strat, exclusive_orders=False, cash=10000)
    output = bt.run()
    
    # Convert the relevant output fields to a dictionary
    result_dict = {
        "Start": str(output['Start']),
        "End": str(output['End']),
        "Duration": str(output['Duration']),
        "Exposure Time [%]": output['Exposure Time [%]'],
        "Equity Final [$]": output['Equity Final [$]'],
        "Equity Peak [$]": output['Equity Peak [$]'],
        "Return [%]": output['Return [%]'],
        "Buy & Hold Return [%]": output['Buy & Hold Return [%]'],
        "Return (Ann.) [%]": output['Return (Ann.) [%]'],
        "Volatility (Ann.) [%]": output['Volatility (Ann.) [%]'],
        "Sharpe Ratio": output['Sharpe Ratio'],
        "Sortino Ratio": output['Sortino Ratio'],
        "Calmar Ratio": output['Calmar Ratio'],
        "Max. Drawdown [%]": output['Max. Drawdown [%]'],
        "Avg. Drawdown [%]": output['Avg. Drawdown [%]'],
        "Max. Drawdown Duration": str(output['Max. Drawdown Duration']),
        "Avg. Drawdown Duration": str(output['Avg. Drawdown Duration']),
        "# Trades": output['# Trades'],
        "Win Rate [%]": output['Win Rate [%]'],
        "Best Trade [%]": output['Best Trade [%]'],
        "Worst Trade [%]": output['Worst Trade [%]'],
        "Avg. Trade [%]": output['Avg. Trade [%]'],
        "Max. Trade Duration": str(output['Max. Trade Duration']),
        "Avg. Trade Duration": str(output['Avg. Trade Duration']),
        "Profit Factor": output['Profit Factor'],
        "Expectancy [%]": output['Expectancy [%]'],
        "SQN": output['SQN'],
    }
    return result_dict


@csrf_exempt
def candlesticks_bot(request, dataframe, backtest_period):
    async def inner_candlesticks():
        try:
            data = json.loads(request.body)

             # Access the boolean values
            engulfing = data.get('engulfing', False)
            pinbar = data.get('pinbar', False)
            morningStar = data.get('morningStar', False)
            threeWhiteSoldiers = data.get('threeWhiteSoldiers', False)
            dojiStar = data.get('dojiStar', False)
            methods = data.get('methods', False)

            result = await handle_api_request_candlesticks(engulfing=engulfing, 
            pinbar=pinbar, morning_star=morningStar, three_white_soldiers=threeWhiteSoldiers, 
            doji_star=dojiStar, methods=methods, dataframe=dataframe, backtest_period=backtest_period)
            return JsonResponse({'Output': result})
        except Exception as e:
            return JsonResponse({'Error': str(e)})

    # Run the asynchronous code using the event loop
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(inner_candlesticks())


def check_moving_averages_for_buy(df, range, ma1_type, ma1, ma2_type, ma2):
    first_ma = f'{ma1_type}_{ma1}'
    second_ma = f'{ma2_type}_{ma2}'
    past_10_rows = df[[second_ma, first_ma]].tail(range)
    past_10_rows['Converge'] = past_10_rows[second_ma] < past_10_rows[first_ma]
    past = past_10_rows.tail(1)['Converge'].values[0]
    second_last_row = past_10_rows['Converge'].iloc[-2]
    print(past_10_rows)
    if past == False and second_last_row == True:
                # print('True')
        return True
    else:
                # print('False')
        return False


def check_moving_averages_for_sell(df, range, ma1_type, ma1, ma2_type, ma2):
    first_ma = f'{ma1_type}_{ma1}'
    second_ma = f'{ma2_type}_{ma2}'
    past_10_rows = df[[second_ma, first_ma]].tail(range)
    past_10_rows['Diverge'] = past_10_rows[second_ma] > past_10_rows[first_ma]
    past = past_10_rows.tail(1)['Diverge'].values[0]
    second_last_row = past_10_rows['Diverge'].iloc[-2]
    # print(past)
    print(past_10_rows)
    if past == False and second_last_row == True:
        # print('True')
        return True
    else:
        # print('False')
        return False
    

@csrf_exempt
def moving_average(df, ma1_type, ma1, ma2_type, ma2):
        range = 2
        # Higher MA
        first_ma = f'{ma1_type}_{ma1}'
        second_ma = f'{ma2_type}_{ma2}'
        if ma1_type == 'SMA':
            df[first_ma] = ta.sma(df['Close'], length=int(ma1))
        else:
            df[first_ma] = ta.ema(df['Close'], length=int(ma1))
        # Lower MA
        if ma2_type == 'SMA':
            df[second_ma] = ta.sma(df['Close'], length=int(ma2))
        else:
            df[second_ma] = ta.ema(df['Close'], length=int(ma2))
        # 1 represents 'BUY'
        # -1 represents 'SELL'
        # 0 represents 'DO NOTHING'

        # already_sell = None
        # already_buy = None
        # try:
        #     already_sell = mt.positions_get()[0]._asdict()['type'] == 1
        # except:
        #     pass
        
        # try:
        #     already_buy = mt.positions_get()[0]._asdict()['type'] == 0
        # except:
        #     pass

        if df.tail(1)[second_ma].values[0] > df.tail(1)[first_ma].values[0]:
        
            if check_moving_averages_for_buy(df=df, range=range, ma1_type=ma1_type, ma1=ma1, ma2_type=ma2_type, ma2=ma2):
                # if already_sell:
                # close_order(ticker, lot_size, buy_order_type, buy_price) # NB
                # time.sleep(1)
                return 1
            return 0
               
        elif df.tail(1)[second_ma].values[0] < df.tail(1)[first_ma].values[0]:
            # print('2')
            # if open_positions is not None:
            #     print('7.0')
            #     close_order(ticker, lot_size, sell_order_type,  sell_price)
            #     print('7')
            # position = 'sell'
            if check_moving_averages_for_sell(df=df, range=range, ma1_type=ma1_type, ma1=ma1, ma2_type=ma2_type, ma2=ma2):
                # if already_buy:
                    # close_order(ticker, lot_size, sell_order_type, sell_price)
                    # time.sleep(1)
                # create_order(ticker, lot_size, sell_order_type, sell_price, sell_sl, sell_tp)
                return -1
            return 0
        else:
            return 0


@csrf_exempt
def api_call(request, asset): 
    # return JsonResponse({"message": "API Call Works!"})  
    # timeframe = timeframe.lower()
    test_variable = MovingAverageBot.objects.all()[0]
    ma1_type = str(test_variable.ma1_type)
    ma2_type = str(test_variable.ma2_type)
    ma1 = str(test_variable.ma1)
    ma2 = str(test_variable.ma2)
    try:
        end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        # Calculate the date 30 days ago from the current day
        start_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")

        # Download data using the calculated dates
        forex_asset = f"{asset}=X"
        data = yf.download(forex_asset, start=start_date, end=end_date, interval="15m")

        moving_average_output = moving_average(df=data, ma1_type=ma1_type, ma1=ma1, ma2_type=ma2_type, ma2=ma2)
            
        output = f'{moving_average_output}'
        return JsonResponse({'message': output})

    except Exception as e:
        return JsonResponse({'message': f'Error: {e}'})


@csrf_exempt
def api_test(request):
    return JsonResponse({"message": "Hello World"})


@csrf_exempt
def new_test(request):
    return JsonResponse({"message": "Heyo!"})


@csrf_exempt
def download_mq4_file(request):
    # Replace with the actual path to your .mq4 file
    location = './moving-average-bot.mq4'
    file_location = os.path.join(os.path.dirname(os.path.realpath(__file__)), location)

    try:
        with open(file_location, 'r') as f:
            file_data = f.read()

        # Create an HTTP response with the file content
        response = HttpResponse(file_data, content_type='application/vnd.ms-excel')
        return response
        response['Content-Disposition'] = 'attachment; filename="bot.mq4"'

        return response
    except FileNotFoundError:
        # Handle file not exist case
        return HttpResponseNotFound('<h1>File not found</h1>')


@csrf_exempt
def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        return {"error": f"Error in encoding image function: {e}"}


@csrf_exempt
def analyse_image(image_data):
    try:
        # OpenAI API Key
        api_key = os.environ.get('OPENAI_API_KEY', '')

        # Getting the base64 string
        base64_image = base64.b64encode(image_data).decode('utf-8')

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please give a technical analysis of this image with at most 2 paragraphs in your response."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        
        json_data = response.json()
        final_response = json_data['choices'][0]['message']['content']
        return final_response

    except Exception as e:
        return {"error": f"Error occurred in analyse image function: {e}"}

@csrf_exempt
def process_image(request):
    if request.method == 'POST':
        try:
            # Get the image data from the request
            data = json.loads(request.body)
            image_base64 = data.get('imageBase64', '')

            # Decode the base64 image data
            image_data = base64.b64decode(image_base64.encode('utf-8'))

            # Perform GPT-4 Vision processing
            analysed_image = analyse_image(image_data)

            return JsonResponse({"status": "success", "result": analysed_image})
        except Exception as e:
            return JsonResponse({"status": "error", "error": str(e)})

    return JsonResponse({"status": "error", "error": "Invalid request method"})
# {status: 'error', error: 'embedded null byte'}


@csrf_exempt
def chosen_models(request, user_email, magic_number):
    try:
        if request.method == 'POST':
            # Decode the bytes to a string
            data_str = request.body.decode('utf-8')
            data = json.loads(data_str)

            today = timezone.localtime(timezone.now()).date()

            # Find the dictionary in the list
            dict_in_list = next((item for item in data if isinstance(item, dict)), None)

            new_model = Bot(username=user_email, magic_number=magic_number, parameters=data, time_saved=today)
            new_model.save()
            return JsonResponse({"message": f"{dict_in_list} with params: {user_email} and {magic_number}"})
        else:
            return JsonResponse({"message": "invalid request method"})
    except Exception as e:
        return JsonResponse({"Error": f"{e}"})



def check_json_in_list(lst):
    
    for item in lst:
        try:
            json.JSONDecoder().decode(item)
            return item
        except json.JSONDecodeError:
            pass
    return None        


@csrf_exempt
def check_model_data(request, user_email, magic_number):
    model_data = Bot.objects.filter(username=user_email, magic_number=magic_number).first()
    return JsonResponse({"output": f"{model_data}"})