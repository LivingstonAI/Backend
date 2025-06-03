from django.core.exceptions import ObjectDoesNotExist
from django.shortcuts import get_object_or_404
from django.http import JsonResponse, HttpResponse, HttpResponseNotFound
from django.db.models import Avg, Count, Sum, Case, When, F
from django.db.models.functions import ExtractWeek, ExtractMonth, ExtractYear
from django.contrib.auth import authenticate, login
from django.views.decorators.csrf import csrf_exempt
from django.utils.timezone import now
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
from datetime import time as datetime_time
import time
import os
import http.client
import urllib.parse
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, resample_apply
import asyncio
from backtesting.test import SMA, GOOG, EURUSD
import pandas as pd
# import patch_pandas_ta
import numpy as np
np.NaN = np.nan

import pandas_ta as ta
# import MetaTrader5 as mt
from datetime import datetime
# from matplotlib import pyplot as plt
import yfinance as yf
import base64
import requests
import ast
import bokeh
from bokeh.io import export_png
from bokeh.plotting import output_file, save
from bokeh.embed import file_html
from bokeh.resources import CDN
from bokeh.embed import json_item
from asgiref.sync import sync_to_async
# import cv2
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
import mplfinance as mpf
from scipy.signal import argrelextrema, find_peaks
from sklearn.neighbors import KernelDensity
import pytz
import openai
# from openai import OpenAI
from django.utils import timezone
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import cot_reports as cot
import seaborn as sns
import io
from twilio.rest import Client
import zipfile
import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow
import pandas as pd
import re
from typing import Tuple
import numpy as np

import matplotlib.pyplot as plt
import datetime
import yfinance as yf
import mplfinance as mpf
import http.client
import urllib.parse
import urllib.request
import base64
import requests

import logging
from django.views.decorators.http import require_http_methods



from PIL import Image
import io

from dataclasses import dataclass



# Comment
# current_hour = datetime.datetime.now().time().hour



scheduler = BackgroundScheduler()
scheduler.start()


def new_york_session():
    now = datetime.now(pytz.timezone('America/New_York')).time()
    return datetime_time(8, 0) <= now <= datetime_time(17, 0)

def london_session():
    now = datetime.now(pytz.timezone('Europe/London')).time()
    return datetime_time(8, 0) <= now <= datetime_time(17, 0)

def asian_session():
    tokyo_now = datetime.now(pytz.timezone('Asia/Tokyo')).time()
    hong_kong_now = datetime.now(pytz.timezone('Asia/Hong_Kong')).time()
    return (datetime_time(8, 0) <= tokyo_now <= datetime_time(17, 0)) or (datetime_time(8, 0) <= hong_kong_now <= datetime_time(17, 0))


@csrf_exempt
def zinaida_feedback_form(request):
    try:

        if request.method == "POST":
            try:
                data = json.loads(request.body)
                feedback = data.get("feedback")
                if not feedback:
                    return JsonResponse({"message": "Feedback cannot be empty."}, status=400)
                
                FeedbackForm.objects.create(feedback=feedback)
                return JsonResponse({"message": "Feedback submitted successfully."}, status=201)
            except Exception as e:
                return JsonResponse({"message": f"An error occurred: {str(e)}"}, status=500)
        return JsonResponse({"message": "Invalid request method."}, status=405)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return JsonResponse({"message": f"An error occurred: {str(e)}"}, status=500)



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


# def get_openai_key(request):
#     return JsonResponse({'OPENAI_API_KEY': os.environ['OPENAI_API_KEY']})


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
    # @csrf_exempt
    def post(self, request, *args, **kwargs):
        global email_of_user
        try:
            email = request.data.get('email')
            password = request.data.get('password')
            test_user = User.objects.get(email=email)
        
            user = authenticate(request, email=email, password=password)
            
            if user is not None:
                email_of_user = email
                login(request, user)
                return Response({'message': 'Login successful', 'email': email}, status=status.HTTP_200_OK)
            else:
                return Response({'message': 'Invalid login credentials'}, status=status.HTTP_401_UNAUTHORIZED)
        except User.DoesNotExist:
            return Response({'message': 'No account with that email'}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({'message': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


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


# Set the OpenAI API key globally
openai.api_key = os.environ['OPENAI_API_KEY']

def chat_gpt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message['content'].strip()



@csrf_exempt
def set_daily_brief_assets(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            selected_assets = data.get('assets', [])

            # Clear existing assets and add the new ones
            DailyBriefAssets.objects.all().delete()
            for asset in selected_assets:
                DailyBriefAssets.objects.create(asset=asset)
            
            return JsonResponse({"message": "Assets updated successfully!"}, status=200)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
    return JsonResponse({"error": "Invalid request method."}, status=400)


@csrf_exempt
def daily_brief(request):
    if request.method == 'POST':
        try:
            update_daily_brief()
            return JsonResponse({'message': 'Daily Brief Updated Successfully!'})
        except Exception as e:
            return JsonResponse({'message': f'Error Occurred in Daily Brief Function: {e}'})
    else:
        return JsonResponse({'message': 'Invalid request method'}, status=405)

def update_daily_brief():
    try:
        # Fetch all assets from the DailyBriefAssets model
        user_assets = DailyBriefAssets.objects.all()
        
        if not user_assets.exists():
            raise ValueError("No assets found in the DailyBriefAssets model.")

        # Retrieve all asset names into a list
        currency_list = [asset.asset for asset in user_assets]
        
        if not currency_list:
            raise ValueError("Currency list is empty.")

        # Clear all previous entries in the dailyBrief table
        dailyBrief.objects.all().delete()

        news_data_list = []
        model_replies_list = []
        
        # Establish a connection to the API
        conn = http.client.HTTPSConnection('api.marketaux.com')

        # Define query parameters
        params_template = {
            'api_token': 'xH2KZ1sYqHmNRpfBVfb9C1BbItHMtlRIdZQoRlYw',
            'language': 'en',
            'limit': 50,
        }

        for asset in currency_list:
            try:
                # Update the symbol in the query parameters
                params = params_template.copy()
                params['symbols'] = asset

                # Send a GET request
                conn.request('GET', '/v1/news/all?{}'.format(urllib.parse.urlencode(params)))

                # Get the response
                res = conn.getresponse()
                data = res.read()

                # Parse the JSON response
                news_data = json.loads(data.decode('utf-8'))

                # Validate the news data
                if 'data' not in news_data or not news_data['data']:
                    raise ValueError(f"No news data available for asset: {asset}")

                # Iterate through the news articles and save specific fields to the database
                for article in news_data['data']:
                    title = article.get('title', '')
                    description = article.get('description', '')
                    source = article.get('source', '')
                    url = article.get('url', '')
                    highlights = (
                        article['entities'][0].get('highlights', '') if article.get('entities') else ''
                    )

                    # Create a dictionary with the specific fields
                    news_entry_data = {
                        'title': title,
                        'description': description,
                        'source': source,
                        'url': url,
                        'highlights': highlights,
                    }
                    news_data_list.append(news_entry_data)

                # Generate a summary using Livingston (or your GPT model)
                livingston_response = chat_gpt(
                    f'Provide me a fundamental data summary of the news data (in paragraph format) for this asset as if you were a professional trader and analyst: {asset}\nWith this news data for the asset: {news_data_list}'
                )
                model_replies_list.append(livingston_response)

                # Get the current date and time
                now = timezone.now()

                # Save the new daily brief entry for this asset
                daily_brief = dailyBrief(asset=asset, summary=livingston_response, last_update=now)
                daily_brief.save()

            except Exception as e:
                print(f"Error processing asset {asset}: {e}")
                # Log the error but continue processing other assets
                return JsonResponse({'error': f'Error Occurred In Update DB Function: {e}'})

        # Return a success message
        return JsonResponse({'message': 'Daily brief successfully updated.'})

    except Exception as e:
        print(f"Exception occurred in update_daily_brief function: {e}")
        # Corrected JSON response
        return JsonResponse({'message': f"Exception occurred in update_daily_brief function: {e}"})


def get_openai_key(request):
    return JsonResponse({'OPENAI_API_KEY': os.environ['OPENAI_API_KEY']})



@csrf_exempt
def reflections_summary(request, asset):
    try:
        asset = asset.upper()
        reflections_list = []
        asset_reflections = Trade.objects.filter(asset=asset)
        for entry in asset_reflections:
            reflections_list.append(entry.reflection)
        livingston_reflections_summary = chat_gpt(f'Please provide me a short one paragraph summary of my reflections for {asset}. Here is the data: {reflections_list}')
        return JsonResponse({'message': f'{livingston_reflections_summary}'})
    except Exception as e:
        print(f'Error occured in reflections_summary function: {e}')
        return JsonResponse({'message': f'Error occured in reflections_summary function: {e}'})
        

# Schedule the update_daily_brief function to run every hour
scheduler.add_job(
    update_daily_brief,
    trigger=IntervalTrigger(hours=1),
    id='update_daily_brief_job',
    name='Update daily brief every hour',
    replace_existing=True
)




@csrf_exempt
def fetch_daily_brief_data(request):
    if request.method == 'GET':
        daily_briefs = dailyBrief.objects.all().order_by('-last_update')
        data = [
            {
                'asset': brief.asset,
                'summary': brief.summary,
                'last_update': brief.last_update,
            } for brief in daily_briefs
        ]
        return JsonResponse(data, status=200, safe=False)


def save_news_data(assets, user_email):
    try:
        news = News.objects.filter(user_email=user_email).delete()
    except:
        pass
    today = timezone.localtime(timezone.now()).date()
    # List of assets to fetch news data for
    assets_to_fetch = assets

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
                user_email=user_email,
                symbol=asset,  # Set the symbol to the current asset
                data=news_entry_data,  # Store the specific fields as JSON data
                day_created=today,  # Use the current datetime as the day_created value
            )
            news_entry.save()
    return JsonResponse({'message': news_data})


@csrf_exempt
def fetch_user_news_data(request, user_email):
    try:
        # # Check if news data for the current day already exists
        today = timezone.localtime(timezone.now()).date()

        # # Check if news data for the current day already exists
        # existing_news = News.objects.filter(user_email=user_email, day_created=today)
        
        # if not existing_news.exists():
        #     # If data for the current day exists, return a message indicating it
        #     save_news_data()
            # return JsonResponse({'message': 'News data for today already exists.'})
        # else:
            # If data for the current day doesn't exist, fetch and save news data
            # return JsonResponse({'message': f'News data for today does not exist {str(today)}'})
        
        # Fetch all news data without using serializers
        news_objects = News.objects.filter(user_email=user_email)
        
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
    except Exception as e:
        return JsonResponse({"news_data": f"no current news data with exception: {e}"})
    


@csrf_exempt
def update_news_data(request, user_email):
    if request.method == 'POST':
        # Retrieve the array of currencies from the request body
        data = request.POST  # For form-encoded data
        # For JSON data, use request.body and decode it
        # Example for JSON data:
        json_data = json.loads(request.body)
        currencies = json_data.get('currencies', [])

        # Process the currencies array as needed
        # Perform preprocessing or any other operations here
        
        
        # Example currency list: ['EURUSD', 'GBPUSD', 'USDJPY', 'EURGBP']

        save_news_data(currencies, user_email)
        # Send back a JSON response indicating success
        return JsonResponse({'message': f'Data is: {currencies} with type: {type(currencies)}'})

    # Handle other HTTP methods or invalid requests
    return JsonResponse({'message': 'Invalid request'}, status=400)    

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

    return_plot = False
        
    
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
        return_plot = True
    elif dataframe == '1D':
        df_to_use = './XAUUSD1D.csv'
        return_plot = True
    
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

    if return_plot:

        p = bt.plot()
        
        item = json_item(p, "myplot")
        # print(item)
        
        plot_json = json.dumps(item)
    else:
        plot_json = {}

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
    return result_dict, plot_json


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

    return_plot = False

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
        return_plot = True
    elif dataframe == '1D':
        df_to_use = './XAUUSD1D.csv'
        return_plot = True
    
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

    df[upper_band] = current_close[upper_band]
    df[middle_band] = current_close[middle_band]
    df[bottom_band] = current_close[bottom_band]
    length = int(len(df) * start)
    second_length = int(len(df) * end)
    bt = Backtest(df[length:second_length], BBands, exclusive_orders=False, cash=10000)
    output = bt.run()

    if return_plot:

        p = bt.plot()
        
        item = json_item(p, "myplot")
        # print(item)
        
        plot_json = json.dumps(item)
    else:
        plot_json = {}

    # image = bt.plot()
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
    return result_dict, plot_json


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
    async def inner_rsi():
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

    p = bt.plot()

    # Convert the plot to HTML
    html = file_html(p, CDN, "my plot")
    
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
    return result_dict, html


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
        # output = f"{data}"
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
def download_mq4_file(request, bot):
    # Replace with the actual path to your .mq4 file
    if bot == 'trading-model':
        location = './trading-model.ex5'
    elif bot == 'risk-bot':
        location = './Risk-Bot-V2.ex5'
    file_location = os.path.join(os.path.dirname(os.path.realpath(__file__)), location)

    try:
        with open(file_location, 'rb') as f:
            file_data = f.read()

        # Create an HTTP response with the file content
        response = HttpResponse(file_data, content_type='application/octet-stream')
        response['Content-Disposition'] = 'attachment; filename="risk-bot.ex5"'
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
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please give a technical analysis of this image (of a trading chart)."
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
            "max_tokens": 1000
        }


        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        
        json_data = response.json()
        final_response = json_data['choices'][0]['message']['content']
        return final_response

    except Exception as e:
        print(f"Error occurred in analyse image function: {e}")
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
            print(f'error: {e}')
            return JsonResponse({"status": "error", "error": str(e)})
    
    print("Invalid request method")
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
def run_bot(request, user_email, magic_number, asset):
    model_data = Bot.objects.filter(username=user_email, magic_number=magic_number).first()
    # model_parameters = list(model_data.parameters)
    model_parameters = ast.literal_eval(model_data.parameters)

    end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    # Calculate the date 30 days ago from the current day
    start_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")

    # Download data using the calculated dates
    forex_asset = f"{asset}=X"
    data = yf.download(forex_asset, start=start_date, end=end_date, interval="15m")

    output = trading_bot(df=data, params=model_parameters)

    # output = f'{data}'
    # Find the dictionary in the list
    dict_in_list = next((item for item in model_parameters if isinstance(item, dict)), None)

    return JsonResponse({"message": f"{output}"})


def trading_bot(df, params):

    trader_params = params

    # Find the dictionary in the list
    # dict_in_list = next((item for item in data if isinstance(item, dict)), None)
    
    dict_in_list = next((item for item in trader_params if isinstance(item, dict)), None)

    bbands_length = ''
    bbands_std = ''

    ma1_type = ''
    ma2_type = ''
    ma1 = ''
    ma2 = ''

    rsi_period = ''
    rsi_overbought = ''
    rsi_oversold = ''

    df['EMA_50'] = ta.ema(df['Close'], length=50)
    df['SMA_200'] = ta.sma(df['Close'], length=200)

    if 'BBands' in trader_params:
        bbands_length = dict_in_list['bbandsLength']
        bbands_std = dict_in_list['bbandsStd']

    elif 'Moving Averages' in trader_params:
        ma1_type = dict_in_list['ma1Type']
        ma1 = dict_in_list['ma1']
        ma2_type = dict_in_list['ma2Type']
        ma2 = dict_in_list['ma2']
    
    elif 'Relative Strength Index (RSI)' in trader_params:
        rsi_period = dict_in_list['rsiPeriod']
        rsi_overbought = dict_in_list['rsiOverbought']
        rsi_oversold = dict_in_list['rsiOversold']
    

    def bullish_engulfing(df):
        df_test = df.tail(7)
        df_test = df_test.drop_duplicates()
        test_size = len(df_test)
        num_engulfing = 0

        for i in range(test_size-1):
            first_candle = df_test.iloc[i-1]
            second_candle = df_test.iloc[i-2]
            third_candle  = df_test.iloc[i-3]
            fourth_candle = df_test.iloc[i-4]
            try:
                fifth_candle = df_test.iloc[i-5]
            except:
                print(df)
                print('')
                print(df_test)
            second_test = first_candle.Close > second_candle.Open

            if is_bearish_candle(second_candle) and is_bullish_candle(first_candle) and second_test == True and is_bearish_run(fifth_candle, fourth_candle, third_candle, second_candle):
                num_engulfing += 1

                # print('Bullish Engulfing')
                if df['EMA_50'].iloc[-1] > df['SMA_200'].iloc[-1]:

                    # Set the style of the plot
                    df.index = pd.to_datetime(df.index)
                    # style = mpf.make_mpf_style(base_mpf_style='classic')

                    # Create the figure object without plotting
                    # fig, axes = mpf.plot(df.tail(candlestick_backtrack), type='candle', volume=True, returnfig=True, style=style)
                    # plt.close(fig)
                    # Save the figure to a file
                    # fig.savefig('candlestick_chart.png')
                    # if self.position:
                    #     self.position.close()

                    # if process_image(self.path) == 2:
                    #     price = self.data.Close[-1]
                    #     gain_amount = self.reward_percentage * self.equity
                    #     risk_amount = self.risk_percentage * self.equity
                    #     tp_level = price + self.reward_percentage
                    #     sl_level = price - self.risk_percentage
                    #     levels = get_fibonacci_levels(df=df.tail(75), trend='uptrend')
                    #     thirty_eight_retracement = levels[2]
                    #     sixty_one8_retracement = levels[4]
                    #     # if thirty_eight_retracement <= price <= sixty_one8_retracement:
                    #     # self.position.close()
                    #     self.buy(tp=tp_level, sl=sl_level)
                    # create_order(ticker, lot_size, buy_order_type, buy_price, buy_sl, buy_tp)
                    return 1
            # break


    def bearish_engulfing(df):
        df_test = df.tail(7)
        df_test = df_test.drop_duplicates()
        test_size = len(df_test)
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
                num_engulfing += 1
                # print('Bearish Engulfing')
                # price = self.data.Close[-1]

                if df['EMA_50'].iloc[-1] < df['SMA_200'].iloc[-1]:
                    df.index = pd.to_datetime(df.index)
                    # style = mpf.make_mpf_style(base_mpf_style='classic')

                    # Create the figure object without plotting
                    # fig, axes = mpf.plot(df.tail(self.candlestick_backtrack), type='candle', volume=True, returnfig=True, style=style)
                    # plt.close(fig)
                    # Save the figure to a file
                    # fig.savefig('candlestick_chart.png')

                    # if self.position:
                    #     self.position.close()
                        # pass
                    # if process_image(self.path) == 0:
                    #     gain_amount = self.reward_percentage * self.equity
                    #     risk_amount = self.risk_percentage * self.equity
                    #     tp_level = price - self.reward_percentage
                    #     sl_level = price + self.risk_percentage
                    #     levels = get_fibonacci_levels(df=df.tail(75), trend='downtrend')
                    #     thirty_eight_retracement = levels[2]
                    #     sixty_one8_retracement = levels[4]
                    #     # if thirty_eight_retracement <= price <= sixty_one8_retracement:
                    #     # self.position.close()
                    #     self.sell(tp=tp_level, sl=sl_level)
                    # create_order(ticker, lot_size, sell_order_type, sell_price, sell_sl, sell_tp)
                    return -1
            # break


    def shooting_star(df):
        # print('')
        dataframe = df.drop_duplicates()
        df = df.tail(5)
        df = df.drop_duplicates()
        test_size = len(df)
        num_shooting_stars = 0
        bullish_shooting_stars = 0

        for i in range((test_size-1)-3):
            first_prev_candle = df.iloc[i]
            second_prev_candle = df.iloc[i+1]
            third_prev_candle = df.iloc[i+2]
            prev_candle = df.iloc[i+3]
            testing_candle = df.iloc[i+4]
            if is_bullish_run(first_prev_candle, second_prev_candle, third_prev_candle, prev_candle):
                test= abs(testing_candle.High - testing_candle.Close)
                if 2 < test < 2.1:
                    num_shooting_stars += 1

                    dataframe.index = pd.to_datetime(dataframe.index)
                    # style = mpf.make_mpf_style(base_mpf_style='classic')

                    # Create the figure object without plotting
                    # fig, axes = mpf.plot(dataframe.tail(75), type='candle', volume=True, returnfig=True, style=style)
                    # plt.close(fig)
                    # Save the figure to a file
                    # fig.savefig('candlestick_chart.png')

                    # print('bearish shooting star')
                    if df['EMA_50'].iloc[-1] < df['SMA_200'].iloc[-1]:
                        return -1

                        # create_order(ticker, lot_size, sell_order_type, sell_price, sell_sl, sell_tp)

            elif is_bearish_run(first_prev_candle, second_prev_candle, third_prev_candle, prev_candle):

                test = abs(testing_candle.High - testing_candle.Close)
                if test > 2 and test < 2.1:
                    bullish_shooting_stars += 1
                    # print('bullish shooting star')
                    dataframe.index = pd.to_datetime(dataframe.index)
                    # style = mpf.make_mpf_style(base_mpf_style='classic')

                    # Create the figure object without plotting
                    # fig, axes = mpf.plot(dataframe.tail(self.candlestick_backtrack), type='candle', volume=True, returnfig=True, style=style)
                    # plt.close(fig)
                    # Save the figure to a file
                    # fig.savefig('candlestick_chart.png')

                    if df['EMA_50'].iloc[-1] > df['SMA_200'].iloc[-1]:
                        return 1
                        # create_order(ticker, lot_size, buy_order_type, buy_price, buy_sl, buy_tp)


    def three_white_soldiers(df):
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

            if is_bearish_run(first_prev_candle, second_prev_candle, third_prev_candle, prev_candle):
                if testing_candle_2.Close > testing_candle.Close and testing_candle.Close > prev_candle.Close:
                    three_white_soldiers += 1
                    # print('bullish three white soldiers')
                    if df['EMA_50'].iloc[-1] > df['SMA_200'].iloc[-1]:
                        dataframe.index = pd.to_datetime(dataframe.index)
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
                        #     price = self.data.Close[-1]
                        #     gain_amount = self.reward_percentage
                        #     risk_amount = self.risk_percentage
                        #     tp_level = price + self.reward_percentage
                        #     sl_level = price - self.risk_percentage
                        #     levels = get_fibonacci_levels(df=dataframe.tail(75), trend='uptrend')
                        #     thirty_eight_retracement = levels[2]
                        #     sixty_one8_retracement = levels[4]
                            # if thirty_eight_retracement <= prev_candle.Close <= sixty_one8_retracement:
                            # self.position.close()
                        # create_order(ticker, lot_size, buy_order_type, buy_price, buy_sl, buy_tp)
                        return 1
            
                            
            elif is_bullish_run(first_prev_candle, second_prev_candle, third_prev_candle, prev_candle):
                if testing_candle_2.Close < testing_candle.Close and testing_candle.Close < prev_candle.Close:
                    three_black_crows += 1
                    # print('bearish three black crows')
                    if df['EMA_50'].iloc[-1] < df['SMA_200'].iloc[-1]:
                        df.index = pd.to_datetime(df.index)
                        # style = mpf.make_mpf_style(base_mpf_style='classic')

                        # Create the figure object without plotting
                        # fig, axes = mpf.plot(df.tail(self.candlestick_backtrack), type='candle', volume=True, returnfig=True, style=style)
                        # plt.close(fig)
                        # # Save the figure to a file
                        # fig.savefig('candlestick_chart.png')
                        # plt.close(fig)
                        # if self.position:
                        #   self.position.close()
                        # if process_image(self.path) == 0:
                        #     price = self.data.Close[-1]
                        #     gain_amount = self.reward_percentage * self.equity
                        #     risk_amount = self.risk_percentage * self.equity
                        #     tp_level = price - self.reward_percentage
                        #     sl_level = price + self.risk_percentage
                        #     levels = get_fibonacci_levels(df=dataframe.tail(75), trend='downtrend')
                        #     thirty_eight_retracement = levels[2]
                        #     sixty_one8_retracement = levels[4]
                            # if thirty_eight_retracement <= prev_candle.Close <= sixty_one8_retracement:
                            # self.position.close()
                        # create_order(ticker, lot_size, sell_order_type, sell_price, sell_sl, sell_tp)
                        return -1
                

    def doji_star(df):
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
            # price = self.data.Close[-1]

            if is_bullish_run(first_prev_candle, second_prev_candle, third_prev_candle, prev_candle):
                test1 = testing_candle.High - testing_candle.Close
                test2 = testing_candle.Close - testing_candle.Low
                if test1 == test2:
                    bullish_doji += 1
                    # print('bullish doji star')
                    # price = self.data.Close[-1]
                    # gain_amount = self.reward_percentage
                    # risk_amount = self.risk_percentage
                    # tp_level = price - gain_amount
                    # sl_level = price + risk_amount

                    dataframe.index = pd.to_datetime(dataframe.index)
                    # style = mpf.make_mpf_style(base_mpf_style='classic')

                    # Create the figure object without plotting
                    # fig, axes = mpf.plot(dataframe.tail(self.candlestick_backtrack), type='candle', volume=True, returnfig=True, style=style)
                    # plt.close(fig)
                    # Save the figure to a file
                    # fig.savefig('candlestick_chart.png')

                    if df['EMA_50'].iloc[-1] < df['SMA_200'].iloc[-1]:
                        return -1
                        # create_order(ticker, lot_size, sell_order_type, sell_price, sell_sl, sell_tp)
            elif is_bearish_run(first_prev_candle, second_prev_candle, third_prev_candle, prev_candle):
                test1 = testing_candle.Open - testing_candle.Close
                test2 = testing_candle.Close - testing_candle.Low
                if test1 == test2:
                    bearish_doji += 1

                    dataframe.index = pd.to_datetime(dataframe.index)
                    # style = mpf.make_mpf_style(base_mpf_style='classic')

                    # Create the figure object without plotting
                    # fig, axes = mpf.plot(dataframe.tail(self.candlestick_backtrack), type='candle', volume=True, returnfig=True, style=style)
                    # plt.close(fig)
                    # Save the figure to a file
                    # fig.savefig('candlestick_chart.png')

                    if df['EMA_50'].iloc[-1] > df['SMA_200'].iloc[-1]:
                        return 1
                        # create_order(ticker, lot_size, buy_order_type, buy_price, buy_sl, buy_tp)


    def bullish_pinbar(df):
        dataframe = df.drop_duplicates()
        df = df.tail(1)
        df = df.drop_duplicates()
        test_size = len(df)
        num_pin_bars = 0
        # price = self.data.Close[-1]

        for i in range(test_size-1):
            candle = df.iloc[i]
            is_pin_bar = (candle.Close - candle.Low) > 0.05
            if is_pin_bar:
                num_pin_bars += 1
                # print('Bullish Pin Bar')

                dataframe.index = pd.to_datetime(dataframe.index)
                # style = mpf.make_mpf_style(base_mpf_style='classic')

                # Create the figure object without plotting
                # fig, axes = mpf.plot(dataframe.tail(self.candlestick_backtrack), type='candle', volume=True, returnfig=True, style=style)
                # plt.close(fig)
                # Save the figure to a file
                # fig.savefig('candlestick_chart.png')

                if df['EMA_50'].iloc[-1] > df['SMA_200'].iloc[-1]:
                    return 1
                    # create_order(ticker, lot_size, buy_order_type, buy_price, buy_sl, buy_tp)


    def bearish_pinbar(df):
        dataframe = df.drop_duplicates()
        df = df.tail(1)
        df = df.drop_duplicates()
        test_size = len(df)
        num_pin_bars = 0

        for i in range(test_size-1):
            candle = df.iloc[i]
            is_pin_bar = abs(candle.Close - candle.High) <  0.05
            if is_pin_bar:
                num_pin_bars += 1
                # print('Bearish Pin Bar')

                dataframe.index = pd.to_datetime(dataframe.index)
                # style = mpf.make_mpf_style(base_mpf_style='classic')

                # Create the figure object without plotting
                # fig, axes = mpf.plot(dataframe.tail(self.candlestick_backtrack), type='candle', volume=True, returnfig=True, style=style)
                # plt.close(fig)
                # Save the figure to a file
                # fig.savefig('candlestick_chart.png')
                if df['EMA_50'].iloc[-1] < df['SMA_200'].iloc[-1]:
                    return -1
                    # create_order(ticker, lot_size, sell_order_type, sell_price, sell_sl, sell_tp)


    def morning_star(df):
        # print('')
        dataframe = df.drop_duplicates()
        df = df.drop_duplicates()
        df = df.tail(6)
        test_size = len(df)
        morning_stars = 0
        evening_stars = 0
        # price = self.data.Close[-1]

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
                    

                    dataframe.index = pd.to_datetime(dataframe.index)
                    # style = mpf.make_mpf_style(base_mpf_style='classic')

                    # Create the figure object without plotting
                    # fig, axes = mpf.plot(dataframe.tail(self.candlestick_backtrack), type='candle', volume=True, returnfig=True, style=style)

                    # Save the figure to a file
                    # fig.savefig('candlestick_chart.png')
                    # plt.close(fig)
                    if df['EMA_50'].iloc[-1] > df['SMA_200'].iloc[-1]:
                        # levels = get_fibonacci_levels(df=dataframe.tail(75), trend='uptrend')
                        # thirty_eight_retracement = levels[2]
                        # sixty_one8_retracement = levels[4]
                        # if thirty_eight_retracement <= testing_candle.Close <= sixty_one8_retracement:
                        # create_order(ticker, lot_size, buy_order_type, buy_price, buy_sl, buy_tp)
                        return 1

            elif is_bullish_run(first_prev_candle, second_prev_candle, third_prev_candle, prev_candle):
                test = testing_candle.Open - testing_candle.Close
                if testing_candle_2.Close < testing_candle.Close and 0 < test < 2 and testing_candle.Close < prev_candle.Close:
                    evening_stars += 1
                    # print('bearish morning star')

                    dataframe.index = pd.to_datetime(dataframe.index)
                    # style = mpf.make_mpf_style(base_mpf_style='classic')

                    # Create the figure object without plotting
                    # fig, axes = mpf.plot(dataframe.tail(self.candlestick_backtrack), type='candle', volume=True, returnfig=True, style=style)

                    # Save the figure to a file
                    # fig.savefig('candlestick_chart.png')
                    # plt.close(fig)
                    if df['EMA_50'].iloc[-1] < df['SMA_200'].iloc[-1]:
                        # levels = get_fibonacci_levels(df=dataframe.tail(75), trend='downtrend')
                        # thirty_eight_retracement = levels[2]
                        # sixty_one8_retracement = levels[4]
                        # if thirty_eight_retracement <= testing_candle.Close <= sixty_one8_retracement:
                        # create_order(ticker, lot_size, sell_order_type, sell_price, sell_sl, sell_tp)
                        return -1


    def matching(df):
        # print('')
        df = df.drop_duplicates()
        dataframe = df
        df = df.tail(5)
        test_size = len(df)
        matching_lows = 0
        matching_highs = 0

        for i in range(test_size-4):
            first_prev_candle = df.iloc[i]
            second_prev_candle = df.iloc[i+1]
            third_prev_candle = df.iloc[i+2]
            prev_candle = df.iloc[i+3]
            testing_candle = df.iloc[i+4]

            if is_bearish_run(first_prev_candle, second_prev_candle, third_prev_candle, prev_candle):
                if prev_candle.Low == testing_candle.Low and prev_candle.Close == testing_candle.Close:
                    matching_lows += 1
                    # print('matching low')

                    dataframe.index = pd.to_datetime(dataframe.index)
                    # style = mpf.make_mpf_style(base_mpf_style='classic')

                    # Create the figure object without plotting
                    # fig, axes = mpf.plot(dataframe.tail(self.candlestick_backtrack), type='candle', volume=True, returnfig=True, style=style)
                    # plt.close(fig)
                    # Save the figure to a file
                    # fig.savefig('candlestick_chart.png')

                    if df['EMA_50'].iloc[-1] > df['SMA_200'].iloc[-1]:
                        # create_order(ticker, lot_size, buy_order_type, buy_price, buy_sl, buy_tp)
                        return 1

            elif is_bullish_run(first_prev_candle, second_prev_candle, third_prev_candle, prev_candle):
                if prev_candle.High == testing_candle.High and prev_candle.High == testing_candle.High:
                    matching_highs += 1
                    # print('matching high')

                    dataframe.index = pd.to_datetime(dataframe.index)
                    # style = mpf.make_mpf_style(base_mpf_style='classic')

                    # Create the figure object without plotting
                    # fig, axes = mpf.plot(dataframe.tail(self.candlestick_backtrack), type='candle', volume=True, returnfig=True, style=style)
                    # plt.close(fig)
                    # Save the figure to a file
                    # fig.savefig('candlestick_chart.png')

                    if df['EMA_50'].iloc[-1] < df['SMA_200'].iloc[-1]:
                        return -1
                        # create_order(ticker, lot_size, sell_order_type, sell_price, sell_sl, sell_tp)


    def methods(df):
        dataframe = df.drop_duplicates()
        df = df.drop_duplicates()
        df = df.tail(8)
        test_size = len(df)
        rising_methods = 0
        falling_methods = 0
        # price = self.data.Close[-1]

        for i in range(test_size-7):
            first_prev_candle = df.iloc[i]
            second_prev_candle = df.iloc[i+1]
            third_prev_candle = df.iloc[i+2]
            prev_candle = df.iloc[i+3]
            testing_candle = df.iloc[i+4]
            testing_candle_2 = df.iloc[i+5]
            testing_candle_3 = df.iloc[i+6]
            final_candle = df.iloc[7]

            if is_bullish_run(first_prev_candle, second_prev_candle, third_prev_candle, prev_candle) and testing_candle.close < prev_candle.close and is_bearish_run_3(testing_candle, testing_candle_2, testing_candle_3):
                if final_candle.Close > prev_candle.Close:
                    rising_methods += 1
                    # print('rising three methods')

                    dataframe.index = pd.to_datetime(dataframe.index)
                    # style = mpf.make_mpf_style(base_mpf_style='classic')

                    # Create the figure object without plotting
                    # fig, axes = mpf.plot(dataframe.tail(self.candlestick_backtrack), type='candle', volume=True, returnfig=True, style=style)
                    # plt.close(fig)
                    # Save the figure to a file
                    # fig.savefig('candlestick_chart.png')
                    # plt.close(fig)

                    if df['EMA_50'].iloc[-1] > df['SMA_200'].iloc[-1]:
                        # levels = get_fibonacci_levels(df=dataframe.tail(75), trend='uptrend')
                        # thirty_eight_retracement = levels[2]
                        # sixty_one8_retracement = levels[4]
                        # if thirty_eight_retracement <= testing_candle_3.Close <= sixty_one8_retracement:
                        # create_order(ticker, lot_size, buy_order_type, buy_price, buy_sl, buy_tp)
                        return 1

            elif is_bearish_run(first_prev_candle, second_prev_candle, third_prev_candle, prev_candle) and testing_candle.close > prev_candle.close and is_bullish_run_3(testing_candle, testing_candle_2, testing_candle_3):
                if final_candle.Close < prev_candle.Close:
                    falling_methods += 1
                    # print('falling three methods')

                    dataframe.index = pd.to_datetime(dataframe.index)
                    # style = mpf.make_mpf_style(base_mpf_style='classic')

                    # Create the figure object without plotting
                    # fig, axes = mpf.plot(dataframe.tail(self.candlestick_backtrack), type='candle', volume=True, returnfig=True, style=style)
                    # plt.close(fig)
                    # Save the figure to a file
                    # fig.savefig('candlestick_chart.png')
                    # plt.close(fig)

                    if df['EMA_50'].iloc[-1] < df['SMA_200'].iloc[-1]:
                        # if thirty_eight_retracement <= testing_candle_3.Close <= sixty_one8_retracement:
                        # create_order(ticker, lot_size, sell_order_type, sell_price, sell_sl, sell_tp)
                        return -1


    def check_moving_averages_for_buy(df, range):
        past_10_rows = df[['SMA_149', 'SMA_202']].tail(range)
        past_10_rows['Converge'] = past_10_rows['SMA_149'] < past_10_rows['SMA_202']
        past = past_10_rows.tail(1)['Converge'].values[0]
        second_last_row = past_10_rows['Converge'].iloc[-2]
        print(past_10_rows)
        if past == False and second_last_row == True:
                # print('True')
            return True
        else:
                # print('False')
            return False


    def check_moving_averages_for_sell(df, range):
        past_10_rows = df[['SMA_149', 'SMA_202']].tail(range)
        past_10_rows['Diverge'] = past_10_rows['SMA_149'] > past_10_rows['SMA_202']
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


    def moving_average(df):
        # print(f'already_sell is {already_sell}')
        # print(f'Open Positions are {open_positions}')
        # position = ''
        # fema = df['EMA_50'].iloc[-1]
        # tsma = df['SMA_200'].iloc[-1]
        # print(f'df ema 50 is {fema}')
        # print(f'df sma 200 is {tsma}')
        if df.tail(1)['MA_Lower'].values[0] > df.tail(1)['MA_Upper'].values[0]:
            # print('1')
            # if open_positions is not None:
            #     close_order(ticker, lot_size, buy_order_type,  buy_price)
            #     print('4')
            if check_moving_averages_for_buy(df=df, range=range):
                # create_order(ticker, lot_size, buy_order_type, buy_price, buy_sl, buy_tp)
                return 1
        
        elif df.tail(1)['MA_Lower'].values[0] < df.tail(1)['MA_Upper'].values[0]:
            # print('2')
            # if open_positions is not None:
            #     print('7.0')
            #     close_order(ticker, lot_size, sell_order_type,  sell_price)
            #     print('7')
            if check_moving_averages_for_sell(df=df, range=range):
                
                # create_order(ticker, lot_size, sell_order_type, sell_price, sell_sl, sell_tp)
                return -1


    def bbands(df):
        upper_band = f'BBU_{bbands_length}_{float(bbands_std)}'
        lower_band = f'BBL_{bbands_length}_{float(bbands_std)}'
        try:
            if df['Close'].iloc[-1] >= df[upper_band].iloc[-1]:
                # create_order(ticker, lot_size, buy_order_type, buy_price, buy_sl, buy_tp)
                return 1
            
            elif df['Close'].iloc[-1] <= df[lower_band].iloc[-1]:
                # create_order(ticker, lot_size, sell_order_type, sell_price, sell_sl, sell_tp)
                return -1
        except: 
            pass


    def rsi(df):
        if df['RSI'].iloc[-1] > rsi_overbought:
            # create_order(ticker, lot_size, sell_order_type, sell_price, sell_sl, sell_tp)
            return -1
        
        elif df['RSI'].iloc[-1] < rsi_oversold:
            # create_order(ticker, lot_size, buy_order_type, buy_price, buy_sl, buy_tp)
            return 1


    def momentum(df):
        if df['MOM'].iloc[-1] > 80:
            # create_order(ticker, lot_size, sell_order_type, sell_price, sell_sl, sell_tp)
            return -1
        elif df['MOM'].iloc[-1] < 20:
            # create_order(ticker, lot_size, buy_order_type, buy_price, buy_sl, buy_tp)
            return 1

    if ma1_type == 'SMA':
        df['MA_Upper'] = ta.sma(df['Close'], length=ma1)
    elif ma1_type == 'EMA':
        df['MA_Upper'] = ta.ema(df['Close'], length=ma1)
    
    if ma2_type == 'SMA':
        df['MA_Lower'] = ta.sma(df['Close'], length=ma2)
    elif ma2_type == 'EMA':
        df['MA_Lower'] = ta.ema(df['Close'], length=ma2)
    # df['MA_Lower'] = ta.sma(df['Close'], length=ma2)
    # print(df)
    current_close = df['Close']
    current_close = ta.bbands(close=df['Close'], length=bbands_length, std=bbands_std, append=True)
    try:
        upper_band = f'BBU_{bbands_length}_{float(bbands_std)}'
        lower_band = f'BBL_{bbands_length}_{float(bbands_std)}'
        df[upper_band] = current_close[upper_band]
        df[lower_band] = current_close[lower_band]
    except: 
        pass

    df['RSI'] = ta.rsi(df['Close'], length = rsi_period)

    df['MOM'] = ta.mom(df['Close'])

    buy = 0
    sell = 0

    if 'Engulfing' in trader_params:
        bullish_engulfing_output = bullish_engulfing(df=df)
        bearish_engulfing_output = bearish_engulfing(df=df)
        if bullish_engulfing_output == 1:
            buy += 1
            # return bullish_engulfing_output
        elif bullish_engulfing_output == -1:
            sell += 1
        if bearish_engulfing_output == 1:
            buy += 1
            # return bearish_engulfing_output
        elif bearish_engulfing_output == -1:
            sell += 1
    elif 'Three White Soldiers' in trader_params:
        three_white_soldiers_output = three_white_soldiers(df=df)
        if three_white_soldiers_output == 1:
            buy += 1
            # return three_white_soldiers_output
        elif three_white_soldiers_output == -1:
            sell += 1
    elif 'Doji Star' in trader_params:
        doji_star_output = doji_star(df=df)
        if doji_star_output == 1:
            buy += 1
            # return doji_star_output
        elif doji_star_output == -1:
            sell += 1
    elif 'Pin Bar' in trader_params:
        bullish_pinbar_output = bullish_pinbar(df=df)
        bearish_pinbar_output = bearish_pinbar(df=df)
        if bullish_pinbar_output == 1:
            buy += 1
            # return bullish_pinbar_output
        elif bullish_pinbar_output == -1:
            sell += 1
        if bearish_pinbar_output == 1:
            buy += 1
        elif bearish_pinbar_output == -1:
            sell += 1
            # return bearish_pinbar_output
    elif 'Morning Star' in trader_params:
        morning_star_output = morning_star(df=df)
        if morning_star_output == 1:
            buy += 1
            # return morning_star_output
        elif morning_star_output == -1:
            sell += 1
    elif 'Matching' in trader_params:
        matching_output = matching(df=df)
        if matching_output == 1:
            buy += 1
            # return matching_output
        elif matching_output == -1:
            sell += 1
    elif 'Methods' in trader_params:
        methods_output = methods(df=df)
        if methods_output == 1:
            buy += 1
        elif methods_output == -1:
            sell += 1
            # return methods_output
    elif 'Moving Averages' in trader_params:
        ma_output = moving_average(df=df)
        if ma_output == 1:
            buy += 1
        elif ma_output == -1:
            sell += 1
            # return ma_output
    elif 'BBands' in trader_params:
        bbands_output = bbands(df=df)
        if bbands_output == 1:
            buy += 1
        elif bbands_output == -1:
            sell += 1
            # return bbands_output
    elif 'Relative Strength Index (RSI)' in trader_params:
        rsi_output = rsi(df=df)
        if rsi_output == 1:
            buy += 1
            # return rsi_output
        elif rsi_output == -1:
            sell += 1
    elif 'Momentum Trading Bot' in trader_params:
        momentum_output = momentum(df=df)
        if momentum_output == 1:
            buy += 1
        elif momentum_output == -1:
            sell += 1

    temp_dict = {}

    temp_dict['buy'] = buy
    temp_dict['sell'] = sell

    if buy > sell:
        return 1
    elif sell > buy:
        return -1
    else:
        return 0
    


@csrf_exempt
async def handle_api_request_backtest(dataframe, backtest_period, parameters):

    class backtestAgent(Strategy):
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
                # Break statement might not be neccessary here. Think it might be safe to remove.
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


        def three_white_soldiers(self, df):
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
            df = df.tail(9)
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
        

        def all_bots(self, df):
            if 'Momentum Trading Bot' in parameters:
                self.momentum(df=df)
            if 'Engulfing' in parameters:
                self.bullish_engulfing(df=df)
                self.bearish_engulfing(df=df)  
            if 'Pin Bar' in parameters:
                self.bullish_pinbar(df=df)
                self.bearish_pinbar(df=df)
            if 'Three White Soldiers' in parameters:
                self.three_white_soldiers(df=df)
            if 'Doji Star' in parameters:
                self.doji_star(df=df)
            if 'Morning Star' in parameters:
                self.morning_star(df=df)
            if 'Methods' in parameters:
                self.methods(df=df)
            

        def next(self):
            df = pd.DataFrame({'Open': self.data.Open, 'High': self.data.High, 'Low': self.data.Low, 'Close': self.data.Close, 'Volume': self.data.Volume})
            df['MOM'] = ta.mom(df['Close'])
            df['EMA_50'] = ta.ema(df['Close'], length=50)
            df['SMA_200'] = ta.sma(df['Close'], length=200)
            # if not self.position:
            try:
                self.all_bots(df=df)
                # print('Running Backtesting Algorithm...')
            except Exception as e:
                print(f'Exception is {e}')
                pass

    return_plot = False
    

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
        return_plot = True
    elif dataframe == '1D':
        df_to_use = './XAUUSD1D.csv'
        return_plot = True
    

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
    bt = Backtest(df[length:second_length], backtestAgent, exclusive_orders=False, cash=10000)
    output = bt.run()
    
    if return_plot:

        p = bt.plot()
        
        item = json_item(p, "myplot")
        # print(item)
        
        plot_json = json.dumps(item)
    else:
        plot_json = {}

    # Convert the plot to HTML
    # html = file_html(plot, CDN, "backtesting plot")

    
    # plot_json = json_item(plot, "myplot")
    # print(item)
    
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
    return result_dict, plot_json


@csrf_exempt
def run_backtest(request, dataframe, backtest_period):
    dummy_param1 = None
    dummy_param2 = None
    try:
        if request.method == 'POST':
            # Decode the bytes to a string
            data_str = request.body.decode('utf-8')
            data = json.loads(data_str)
            model_parameters = data
            dummy_param1 = data
            async def inner_backtest():
                result = await handle_api_request_backtest(dataframe, backtest_period, model_parameters)
                return JsonResponse({'Output': result})

            # Run the asynchronous code using the event loop
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(inner_backtest())
    except Exception as e:
        return JsonResponse({"Error Occured Here": f'{e}'})


@csrf_exempt
def interest_rates(request):
    api_ninjas_key = 'fhw7p7lWporgmk7eGGdpiQ==ce3O6xofIN88xuH2'
    api_url = 'https://api.api-ninjas.com/v1/interestrate'
    result = ''
    response = requests.get(api_url, headers={'X-Api-Key': api_ninjas_key, 'central_bank_only': 'true'})
    if response.status_code == requests.codes.ok:
        result = response.text
    else:
        result = response.status_code, response.text
    return JsonResponse({'Interest Rates': result})



def is_bearish_candle(candle):
    # Function to check if a candle is a bearish candle
    if candle.Open > candle.Close:
        return True
    return False


def is_bullish_candle(candle):
    # Function to check if a candle is a bullish candle
    if candle.Open < candle.Close:
        return True
    else:
        return False
    
# candle_type = is_bullish_candle(temp_dataset.head(1))
# print(f'Bullish Candle Type is: {candle_type}\n')


def is_bullish_engulfing(data):

    # In this function we take the final 3 candles. We then check if the first two 
    # candles (excluding the latest 3rd one) satisfies the candle that it is a bullish 
    # engulfing.
    candle1 = data.iloc[-3]
    candle2 = data.iloc[-2]

    if is_bearish_candle(candle1) and is_bullish_candle(candle2):
        if candle2.Close > candle1.Open and candle2.Open < candle1.Close:
            return True
        return False
    return False


def is_bearish_engulfing(data):

    # In this function we take the final 3 candles. We then check if the first two 
    # candles (excluding the latest 3rd one) satisfies the candle that it is a bearish
    # engulfing.
    candle1 = data.iloc[-3]
    candle2 = data.iloc[-2]

    if is_bullish_candle(candle1) and is_bearish_candle(candle2):
        if candle2.Close < candle1.Open and candle2.Open > candle1.Close:
            return True
        return False
    return False


def is_morning_star(data):

    # In this function we take the final 3 candles. We then check if the first two 
    # candles (excluding the latest 3rd one) satisfies the candle that it is a morning
    # star.
    candle1 = data.iloc[-4]
    candle2 = data.iloc[-3]
    candle3 = data.iloc[-2]

    if is_bearish_candle(candle1) and is_bullish_candle(candle3):
        if candle3.Close < candle1.Open and candle2.Close < candle3.Close:
            return True
        return False
    return False


def is_evening_star(data):

    # In this function we take the final 3 candles. We then check if the first two 
    # candles (excluding the latest 3rd one) satisfies the candle that it is an evening
    # star.
    candle1 = data.iloc[-4]
    candle2 = data.iloc[-3]
    candle3 = data.iloc[-2]

    if is_bullish_candle(candle1) and is_bearish_candle(candle3):
        if candle3.Close > candle1.Open and candle2.Close > candle3.Close:
            return True
        return False
    return False


def is_three_white_soldiers(data):
     # In this function we take the final 3 candles. We then check if the first two 
    # candles (excluding the latest 3rd one) satisfies the candle that it is a three
    # white soldiers.
    candle1 = data.iloc[-4]
    candle2 = data.iloc[-3]
    candle3 = data.iloc[-2]

    if is_bullish_candle(candle3) and is_bullish_candle(candle2) and is_bullish_candle(candle1):
        if candle3.Close > candle2.Close and candle2.Close > candle1.Close:
            return True
        return False
    return False


def is_three_black_crows(data):
     # In this function we take the final 3 candles. We then check if the first two 
    # candles (excluding the latest 3rd one) satisfies the candle that it is a three
    # white soldiers.
    candle1 = data.iloc[-4]
    candle2 = data.iloc[-3]
    candle3 = data.iloc[-2]

    if is_bearish_candle(candle3) and is_bearish_candle(candle2) and is_bearish_candle(candle1):
        if candle3.Close < candle2.Close and candle2.Close < candle1.Close:
            return True
        return False
    return False


def is_morning_doji_star(data):
    # In this function we take the final 3 candles. We then check if the first two 
    # candles (excluding the latest 3rd one) satisfies the candle that it is a morning
    # doji star.
    candle1 = data.iloc[-4]
    candle2 = data.iloc[-3]
    candle3 = data.iloc[-2]

    if is_bearish_candle(candle1) and is_bullish_candle(candle3) and candle3.Close < candle1.Open:
        if candle2.Open == candle2.Close:
            return True
        return False
    return False


def is_evening_doji_star(data):
    # In this function we take the final 3 candles. We then check if the first two 
    # candles (excluding the latest 3rd one) satisfies the candle that it is a morning
    # doji star.
    candle1 = data.iloc[-4]
    candle2 = data.iloc[-3]
    candle3 = data.iloc[-2]

    if is_bullish_candle(candle1) and is_bearish_candle(candle3) and candle3.Close > candle1.Open:
        if candle2.Open == candle2.Close:
            return True
        return False
    return False


def is_rising_three_methods(data):

    candle1 = data.iloc[-6]
    candle2 = data.iloc[-5]
    candle3 = data.iloc[-4]
    candle4 = data.iloc[-3]
    candle5 = data.iloc[-2]

    if is_bullish_candle(candle1) and is_bearish_candle(candle2) \
    and is_bearish_candle(candle3) and is_bearish_candle(candle4) and is_bullish_candle(candle5):
        if candle2.Close > candle3.Close and candle3.Close > candle4.Close and \
            candle5.Close > candle1.Close:
            return True
        return False
    return False


def is_falling_three_methods(data):

    candle1 = data.iloc[-6]
    candle2 = data.iloc[-5]
    candle3 = data.iloc[-4]
    candle4 = data.iloc[-3]
    candle5 = data.iloc[-2]

    if is_bearish_candle(candle1) and is_bullish_candle(candle2) \
    and is_bullish_candle(candle3) and is_bullish_candle(candle4) and is_bearish_candle(candle5):
        if candle2.Close < candle3.Close and candle3.Close < candle4.Close and \
            candle5.Close < candle1.Close:
            return True
        return False
    return False


def is_hammer(data):

    candle1 = data.iloc[-6]
    candle2 = data.iloc[-5]
    candle3 = data.iloc[-4]
    candle4 = data.iloc[-3]
    candle5 = data.iloc[-2]

    if is_bearish_candle(candle1) and is_bearish_candle(candle2) and is_bullish_candle(candle3) \
    and is_bullish_candle(candle4) and is_bullish_candle(candle5):
        if abs(candle3.Close - candle3.Open) < abs(candle3.Open - candle3.Low): 
            return True
        return False
    return False


def is_hanging_man(data):

    candle1 = data.iloc[-6]
    candle2 = data.iloc[-5]
    candle3 = data.iloc[-4]
    candle4 = data.iloc[-3]
    candle5 = data.iloc[-2]

    if is_bullish_candle(candle1) and is_bullish_candle(candle2) and is_bearish_candle(candle3) \
    and is_bearish_candle(candle4) and is_bearish_candle(candle5):
        if abs(candle3.Open - candle3.Close) < abs(candle3.Close - candle3.Low): 
            return True
        return False
    return False


def is_inverted_hammer(data):

    candle1 = data.iloc[-6]
    candle2 = data.iloc[-5]
    candle3 = data.iloc[-4]
    candle4 = data.iloc[-3]
    candle5 = data.iloc[-2]

    if is_bearish_candle(candle1) and is_bearish_candle(candle2) and is_bullish_candle(candle3) \
    and is_bullish_candle(candle4) and is_bullish_candle(candle5):
        if abs(candle3.Close - candle3.Open) < abs(candle3.High - candle3.Close): 
            return True
        return False
    
    elif is_bearish_candle(candle1) and is_bearish_candle(candle2) and is_bearish_candle(candle3) \
    and is_bullish_candle(candle4) and is_bullish_candle(candle5):
        if abs(candle3.Close - candle3.Open) < abs(candle3.High - candle3.Open): 
            return True
        return False
    
    return False


def is_shooting_star(data):

    candle1 = data.iloc[-6]
    candle2 = data.iloc[-5]
    candle3 = data.iloc[-4]
    candle4 = data.iloc[-3]
    candle5 = data.iloc[-2]

    if is_bullish_candle(candle1) and is_bullish_candle(candle2) and is_bullish_candle(candle3) \
    and is_bearish_candle(candle4) and is_bearish_candle(candle5):
        if abs(candle3.Close - candle3.Open) < abs(candle3.High - candle3.Close): 
            return True
        return False

    elif is_bullish_candle(candle1) and is_bullish_candle(candle2) and is_bearish_candle(candle3) \
    and is_bearish_candle(candle4) and is_bearish_candle(candle5):
        if abs(candle3.Close - candle3.Open) < abs(candle3.High - candle3.Open): 
            return True
        return False
    
    return False


def is_bullish_kicker(data):

    candle1 = data.iloc[-5]
    candle2 = data.iloc[-4]
    candle3 = data.iloc[-3]
    candle4 = data.iloc[-2]

    if is_bearish_candle(candle1) and is_bearish_candle(candle2) and is_bearish_candle(candle3) \
    and is_bullish_candle(candle4):
        if candle4.Close > candle3.Open:
            return True
        return False
    
    return False


def is_bearish_kicker(data):

    candle1 = data.iloc[-5]
    candle2 = data.iloc[-4]
    candle3 = data.iloc[-3]
    candle4 = data.iloc[-2]

    if is_bullish_candle(candle1) and is_bullish_candle(candle2) and is_bullish_candle(candle3) \
    and is_bearish_candle(candle4):
        if candle4.Close < candle3.Open:
            return True
        return False    
    return False


def is_bullish_harami(data):

    candle1 = data.iloc[-3]
    candle2 = data.iloc[-2]

    if is_bearish_candle(candle1) and is_bullish_candle(candle2):
        if candle2.Close < candle1.Open and candle2.Open > candle1.Close:
            return True
        return False
    return False


def is_bearish_harami(data):

    candle1 = data.iloc[-3]
    candle2 = data.iloc[-2]

    if is_bullish_candle(candle1) and is_bearish_candle(candle2):
        if candle2.Close > candle1.Open and candle2.Open < candle1.Close:
            return True
        return False
    return False


def is_bullish_three_line_strike(data):

    candle1 = data.iloc[-5]
    candle2 = data.iloc[-4]
    candle3 = data.iloc[-3]
    candle4 = data.iloc[-2]

    if is_bearish_candle(candle1) and is_bearish_candle(candle2) and is_bearish_candle(candle3) \
    and is_bullish_candle(candle4):
        if candle2.Close < candle1.Close and candle3.Close < candle2.Close and \
        candle4.Close > candle1.Open:
            return True
        return False

    return False


def is_bearish_three_line_strike(data):

    candle1 = data.iloc[-5]
    candle2 = data.iloc[-4]
    candle3 = data.iloc[-3]
    candle4 = data.iloc[-2]

    if is_bullish_candle(candle1) and is_bullish_candle(candle2) and is_bullish_candle(candle3) \
    and is_bearish_candle(candle4):
        if candle2.Close > candle1.Close and candle3.Close > candle2.Close and \
        candle4.Close < candle1.Open:
            return True
        return False

    return False


def moving_average(type, number, data):
    latest_moving_average_value = 0
    if type == 'SMA':
        latest_moving_average_value = ta.sma(data['Close'], length=number)
    elif type == 'EMA':
        latest_moving_average_value = ta.ema(data['Close'], length=number)
    return latest_moving_average_value.iloc[-1]


def bbands(condition, band, data):
    band = band.upper()
    condition = condition.upper()
    bbands = ta.bbands(data['Close'], length=20, std=2)  # By default, it calculates with a length of 20 and std of 2
    upper_band = bbands['BBU_20_2.0'].iloc[-1]
    middle_band = bbands['BBM_20_2.0'].iloc[-1]
    lower_band = bbands['BBL_20_2.0'].iloc[-1]
    current_price = data.Close.iloc[-1]

    if condition == 'LT' and band == 'LOWER':
        if current_price < lower_band:
            return True
        return False
    elif condition == 'LT' and band == 'MIDDLE':
        if current_price < middle_band:
            return True
        return False
    elif condition == 'LT' and band == 'UPPER':
        if current_price < upper_band:
            return True
        return False
    
    elif condition == 'GT' and band == 'LOWER':
        if current_price > lower_band:
            return True
        return False
    elif condition == 'GT' and band == 'MIDDLE':
        if current_price > middle_band:
            return True
        return False
    elif condition == 'GT' and band == 'UPPER':
        if current_price > upper_band:
            return True
        return False
    return False


def momentum(comparison, threshold, data):
    comparison = comparison.upper()
    current_momentum = ta.mom(data['Close']).iloc[-1]
    if comparison == 'ABOVE':
        if current_momentum > threshold:
            return True
        return False
    elif comparison == 'BELOW':
        if current_momentum < threshold:
            return True
        return False
    return False


def rsi(comparison, rsi_level, data):
    comparison = comparison.upper()
    current_rsi = ta.rsi(data['Close']).iloc[-1]
    if comparison == 'ABOVE':
        if current_rsi > rsi_level:
            return True
        return False
    elif comparison == 'BELOW':
        if current_rsi < rsi_level:
            return True
        return False
    return False


# Function to split Dataset By Year 
def split_df(df, start_year, end_year):
    # Convert the index to datetime if it's not already
    df.index = pd.to_datetime(df.index)
    
    # Filter rows based on the specified start and end years
    new_df = df[(df.index.year >= start_year) & (df.index.year <= end_year)]

    return new_df


async def genesys_backest(generated_code, start_year, end_year, chosen_dataset, initial_capital):

    class GenesysBacktest(Strategy):
        def init(self):
            price = self.data.Close
            self.current_equity = 0
            self.true_init_equity = init_capital

        def set_take_profit(self, number, type_of_setting):
            current_equity = self.equity
            # print(f'Current Equity: {current_equity}\n')
            type_of_setting = type_of_setting.upper()
            number = float(number)
            
            # print(f'self.init_equity: {self.init_equity} vs current equity: {current_equity} with diff: {((current_equity - self.init_equity) / self.true_init_equity) * 100}')
            if type_of_setting == 'PERCENTAGE':
                percentage = ((current_equity - self.current_equity) / self.true_init_equity) * 100
                if percentage >= number:
                    self.position.close()
            elif type_of_setting == 'NUMBER':
                difference = current_equity - self.current_equity
                if difference >= number:
                    self.position.close()
        

        def set_stop_loss(self, number, type_of_setting):
            type_of_setting = type_of_setting.upper()
            number = -(float(number))
            current_equity = self.equity
            if type_of_setting == 'PERCENTAGE':
                percentage = ((current_equity - self.current_equity) / self.true_init_equity) * 100
                if percentage <= number:
                    self.position.close()
            elif type_of_setting == 'NUMBER':
                difference = current_equity - self.current_equity
                if difference <= number:
                    self.position.close()
                    

          
        def next(self):
            dataset = pd.DataFrame({'Open': self.data.Open, 'High': self.data.High, 'Low': self.data.Low, 'Close': self.data.Close, 'Volume': self.data.Volume})
            try:
                exec(generated_code)    
            except Exception as e:
                print(f'Exception: {e}')
    try:
        # Query the model asynchronously using sync_to_async
        queryset = await sync_to_async(SaveDataset.objects.all().first)()
        dataset_to_use = f'./{chosen_dataset}'
        df_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataset_to_use)
        df = pd.read_csv(df_path).drop_duplicates()
        df.index = pd.to_datetime(df['Time'].values)
        del df['Time']
        
        # split_queryset = await sync_to_async(SplitDataset.objects.get)()

        start_year = int(start_year)
        end_year = int(end_year)
        new_df = split_df(df, start_year, end_year)
        # print(df)
        
        # init_capital_queryset = await sync_to_async(SetInitCapital.objects.get)()
        # init_capital = float(init_capital_queryset.initial_capital)
        init_capital = initial_capital

        bt = Backtest(new_df, GenesysBacktest,
                exclusive_orders=True, cash=init_capital)

        output = bt.run()

        # return_plot = False

        # if len(new_df) > 5000:
        #     return_plot = True
        # else:
        #     return_plot = False
        try:
            p = bt.plot()
                
            item = json_item(p, "new_plot")
            # print(item)
                
            plot_json = json.dumps(item)
        except Exception as e:
            plot_json = {}
        
        # Convert the relevant output fields to a dictionary
        result_dict = {
            "Start": str(output['Start']),
            "End": str(output['End']),
            "Duration": str(output['Duration']),
            "Exposure Time [%]": str(output['Exposure Time [%]']),
            "Equity Final [$]": str(output['Equity Final [$]']),
            "Equity Peak [$]": str(output['Equity Peak [$]']),
            "Return [%]": str(output['Return [%]']),
            "Buy & Hold Return [%]": str(output['Buy & Hold Return [%]']),
            "Return (Ann.) [%]": str(output['Return (Ann.) [%]']),
            "Volatility (Ann.) [%]": str(output['Volatility (Ann.) [%]']),
            "Sharpe Ratio": str(output['Sharpe Ratio']),
            "Sortino Ratio": str(output['Sortino Ratio']),
            "Calmar Ratio": str(output['Calmar Ratio']),
            "Max. Drawdown [%]": str(output['Max. Drawdown [%]']),
            "Avg. Drawdown [%]": str(output['Avg. Drawdown [%]']),
            "Max. Drawdown Duration": str(output['Max. Drawdown Duration']),
            "Avg. Drawdown Duration": str(output['Avg. Drawdown Duration']),
            "# Trades": str(output['# Trades']),
            "Win Rate [%]": str(output['Win Rate [%]']),
            "Best Trade [%]": str(output['Best Trade [%]']),
            "Worst Trade [%]": str(output['Worst Trade [%]']),
            "Avg. Trade [%]": str(output['Avg. Trade [%]']),
            "Max. Trade Duration": str(output['Max. Trade Duration']),
            "Avg. Trade Duration": str(output['Avg. Trade Duration']),
            "Profit Factor": str(output['Profit Factor']),
            "Expectancy [%]": str(output['Expectancy [%]']),
            # "SQN": output['SQN'],
        }
        return result_dict, plot_json
    except Exception as e:
        return JsonResponse({'error': str(e)})


@csrf_exempt
def genesys(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            generated_code = data.get('generatedCode', '')

            # Execute the generated code

            try:
                async def inner_genesys_backtest():
                    result = await genesys_backest(generated_code, start_year, end_year, chosen_dataset, initial_capital)
                    return JsonResponse({'message': result})

                # Run the asynchronous code using the event loop
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(inner_genesys_backtest())
                    
                # return JsonResponse({'message': f'{result}'})
            except Exception as e:
                return JsonResponse({'error': f'Error executing code: {str(e)}'}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
    else:
        return JsonResponse({'message': 'api-call works!'})

async def run_genesys_backtests():
    """
    Periodically fetches untested backtest models, runs the backtest, and stores the results.
    """
    try:
        # Get all untested models
        untested_models = await sync_to_async(list)(BacktestModels.objects.filter(model_backtested=False))
        
        for model in untested_models:
            try:
                # Run the backtest
                result_dict, plot_json = await genesys_backest(
                    model.generated_code,
                    model.dataset_start,
                    model.dataset_end,
                    model.chosen_dataset,
                    model.initial_capital
                )
                
                # Parse dates
                from datetime import datetime
                start_date = datetime.strptime(result_dict.get("Start"), "%Y-%m-%d %H:%M:%S").date()
                end_date = datetime.strptime(result_dict.get("End"), "%Y-%m-%d %H:%M:%S").date()

                # Create the result object using sync_to_async
                await sync_to_async(BacktestResult.objects.create)(
                    backtest_model=model,
                    start=start_date,
                    end=end_date,
                    duration=result_dict.get("Duration"),
                    exposure_time=float(result_dict.get("Exposure Time [%]", 0)),
                    equity_final=float(result_dict.get("Equity Final [$]", 0)),
                    equity_peak=float(result_dict.get("Equity Peak [$]", 0)),
                    return_percent=float(result_dict.get("Return [%]", 0)),
                    buy_hold_return=float(result_dict.get("Buy & Hold Return [%]", 0)),
                    annual_return=float(result_dict.get("Return (Ann.) [%]", 0)),
                    volatility_annual=float(result_dict.get("Volatility (Ann.) [%]", 0)),
                    sharpe_ratio=float(result_dict.get("Sharpe Ratio", 0)),
                    sortino_ratio=float(result_dict.get("Sortino Ratio", 0)),
                    calmar_ratio=float(result_dict.get("Calmar Ratio", 0)),
                    max_drawdown=float(result_dict.get("Max. Drawdown [%]", 0)),
                    avg_drawdown=float(result_dict.get("Avg. Drawdown [%]", 0)),
                    max_drawdown_duration=result_dict.get("Max. Drawdown Duration"),
                    avg_drawdown_duration=result_dict.get("Avg. Drawdown Duration"),
                    num_trades=int(result_dict.get("# Trades", 0)),
                    win_rate=float(result_dict.get("Win Rate [%]", 0)),
                    best_trade=float(result_dict.get("Best Trade [%]", 0)),
                    worst_trade=float(result_dict.get("Worst Trade [%]", 0)),
                    avg_trade=float(result_dict.get("Avg. Trade [%]", 0)),
                    max_trade_duration=result_dict.get("Max. Trade Duration"),
                    avg_trade_duration=result_dict.get("Avg. Trade Duration"),
                    profit_factor=float(result_dict.get("Profit Factor", 0)),
                    expectancy=float(result_dict.get("Expectancy [%]", 0)),
                    plot_json=plot_json
                )

                # Update model status
                model.model_backtested = True
                await sync_to_async(model.save)()
                
                print(f"Successfully processed backtest for model {model.id}")
                
            except Exception as e:
                print(f"Error processing backtest for model {model.id}: {str(e)}")
                # Consider adding error tracking to your model
                # model.error_message = str(e)
                # await sync_to_async(model.save)()
    
    except Exception as e:
        print(f"Error in run_genesys_backtests: {str(e)}")

        # except Exception as e:
        #     print(f"Error processing backtest for {model}: {e}")
from asgiref.sync import async_to_sync  # Import async_to_sync to call async functions

@csrf_exempt
def trigger_backtest(request):
    try:
        # Create a new event loop for this request
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the async function in this new loop
            loop.run_until_complete(run_genesys_backtests())
            return JsonResponse({"status": "success", "message": "Backtest completed successfully."}, status=200)
        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)}, status=500)
        finally:
            # Always close the loop
            loop.close()
    except Exception as e:
        return JsonResponse({"status": "error", "message": f"Loop error: {str(e)}"}, status=500)


# import asyncio

# Replace your current scheduler code with this

def run_genesys_backtests_wrapper():
    """
    Wrapper function that sets up the event loop and runs the async function
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(run_genesys_backtests())
    finally:
        loop.close()

# Schedule the wrapper function instead of the async function directly
scheduler.add_job(
    run_genesys_backtests_wrapper,
    trigger=IntervalTrigger(minutes=30),
    id='run_genesys_backtests',
    name='Update genesys backtests every 30 minutes',
    replace_existing=True
)


@csrf_exempt
def fetch_backtested_results(request):
    try:
        # Get all backtest results ordered by creation time in descending order
        results = BacktestResult.objects.select_related('backtest_model').order_by('-created_at')
        
        # Group results by backtest model
        grouped_results = {}
        
        for result in results:
            model_id = result.backtest_model.id if result.backtest_model else 'unknown'
            
            if model_id not in grouped_results:
                # If this is the first result for this model, initialize the model info
                if result.backtest_model:
                    grouped_results[model_id] = {
                        'model_info': {
                            'id': result.backtest_model.id,
                            'dataset': result.backtest_model.chosen_dataset,
                            'start_date': result.backtest_model.dataset_start,
                            'end_date': result.backtest_model.dataset_end,
                            'initial_capital': result.backtest_model.initial_capital,
                            'code_snippet': result.backtest_model.generated_code,

                            # 'code_snippet': result.backtest_model.generated_code if len(result.backtest_model.generated_code) > 200 else result.backtest_model.generated_code,
                        },
                        'results': []
                    }
                else:
                    grouped_results[model_id] = {
                        'model_info': {
                            'id': 'unknown',
                            'dataset': 'Unknown',
                            'start_date': 'Unknown',
                            'end_date': 'Unknown',
                            'initial_capital': 0,
                            'code_snippet': 'No code available',
                        },
                        'results': []
                    }
            
            # Add this result to the appropriate group
            grouped_results[model_id]['results'].append({
                'id': result.id,
                'start': result.start,
                'end': result.end,
                'duration': result.duration,
                'exposure_time': result.exposure_time,
                'equity_final': result.equity_final,
                'equity_peak': result.equity_peak,
                'return_percent': result.return_percent,
                'buy_hold_return': result.buy_hold_return,
                'annual_return': result.annual_return,
                'volatility_annual': result.volatility_annual,
                'sharpe_ratio': result.sharpe_ratio,
                'sortino_ratio': result.sortino_ratio,
                'calmar_ratio': result.calmar_ratio,
                'max_drawdown': result.max_drawdown,
                'avg_drawdown': result.avg_drawdown,
                'max_drawdown_duration': result.max_drawdown_duration,
                'avg_drawdown_duration': result.avg_drawdown_duration,
                'num_trades': result.num_trades,
                'win_rate': result.win_rate,
                'best_trade': result.best_trade,
                'worst_trade': result.worst_trade,
                'avg_trade': result.avg_trade,
                'max_trade_duration': result.max_trade_duration,
                'avg_trade_duration': result.avg_trade_duration,
                'profit_factor': result.profit_factor,
                'expectancy': result.expectancy,
                'created_at': result.created_at,
                'has_plot': bool(result.plot_json),
                'plot_json': result.plot_json
            })
        
        # Convert to list for easier handling in React
        results_list = list(grouped_results.values())
        
        return JsonResponse({'status': 'success', 'data': results_list})
    
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)




@csrf_exempt
def delete_backtest_model(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            model_id = data.get('model_id')
            
            if not model_id:
                return JsonResponse({'status': 'error', 'message': 'Model ID is required'})
                
            model = BacktestModels.objects.get(id=model_id)
            # This will also delete all related BacktestResult objects due to CASCADE
            model.delete()
            
            return JsonResponse({'status': 'success', 'message': 'Model and associated results deleted'})
        except BacktestModels.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Model not found'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    
    return JsonResponse({'status': 'error', 'message': 'Only POST method is allowed'})

# @csrf_exempt
# async def test_async_backtest(request):
#     try:
#         # Run the asynchronous backtest function
#         await run_genesys_backtests()
#         return JsonResponse({'message': 'Backtest Successful!'})
#     except Exception as e:
#         return JsonResponse({'message': str(e)})


@csrf_exempt
def save_dataset(request, dataset):
    dataset_to_save = dataset
    try:
        try:
            # If there are no saved objects
            SaveDataset.objects.all().delete()
        except Exception as e:
            pass
        new_dataset = SaveDataset(dataset=dataset_to_save)
        new_dataset.save()

        query_test = SaveDataset.objects.all().first().dataset
        return JsonResponse({'saved-dataset': f'{query_test}'})

    except Exception as e:
        return JsonResponse({'error': f'Error Occured: {e}'})
    

@csrf_exempt
def split_dataset(request):

    try:
        SplitDataset.objects.all().delete()
    except Exception as e:
        pass

    if request.method == 'POST':
        # Parse the request data
        data = json.loads(request.body)
        start_year = data.get('start_year')
        end_year = data.get('end_year')

        # Perform any necessary validation
        # For example, check if start_year and end_year are valid integers

        # Save the start and end years to the database
        split_dataset = SplitDataset.objects.create(
            start_year=start_year,
            end_year=end_year
        )

        # Return a success response
        return JsonResponse({'message': 'Start and end years saved successfully.'})

    # If the request method is not POST, return an error response
    return JsonResponse({'error': 'Only POST requests are allowed.'}, status=405)


@csrf_exempt
def set_init_capital(request):

    try:
        SetInitCapital.objects.all().delete()
    except Exception as e:
        pass

    if request.method == 'POST':
        data = json.loads(request.body)
        initial_capital = data.get('initialCapital')

        # Save initial capital to the database
        SetInitCapital.objects.create(initial_capital=initial_capital)

        return JsonResponse({'message': 'Initial capital saved successfully'})
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)


# def obtain_dataset(asset, interval, num_days):

#     # Calculate the date 30 days ago from the current day
#     start_date = (datetime.now() - timedelta(days=num_days)).strftime("%Y-%m-%d")

#     # Get latest candle
#     end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

#     # Download data using the calculated dates
#     forex_asset = f"{asset}=X"
#     data = yf.download(forex_asset, start=start_date, end=end_date, interval=interval)

#     return data


def generate_trading_image(df):
    df.index = pd.to_datetime(df.index)
    style = mpf.make_mpf_style(base_mpf_style='classic')
    fig, axes = mpf.plot(df, type='line', volume=False, returnfig=True, style=style)
    plt.close(fig)
    output_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'trading_chart.png')
    fig.savefig(output_path)


@csrf_exempt
def test_cnn(request, asset, interval, num_days):
    dataset = obtain_dataset(asset=asset, interval=interval, num_days=num_days)
    # classification = image_classification(data=dataset)
    up = is_uptrend(data=dataset)
    down = is_downtrend(data=dataset)
    ranger = is_ranging_market(data=dataset)
    return JsonResponse({'classification': f'  Uptrend: {up}\nDowntrend: {down}\nRanging Market: {ranger}'})


def image_classification(data):
    generate_trading_image(df=data)
    # print('Loading Model')
    url = "https://us-central1-glowing-road-419608.cloudfunctions.net/function-1"
    # path_to_image = '/candlestick_chart.png'
    path_to_image = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'trading_chart.png')


    with open(path_to_image, 'rb') as image_file:
        # Read the image file and encode it as base64
        image_data = base64.b64encode(image_file.read()).decode('utf-8')

        headers = {'Content-Type': 'application/json'}
        payload = {'data': image_data}

        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()  # Check for HTTP errors

            # Extract the predictions from the response
            response_data = response.json()
            # predictions = response_data['predictions'][0]

            # Find the index of the highest prediction
            # highest_pred_index = predictions.index(max(predictions))

            # Map the index to the corresponding class name
            # predicted_class = class_names[highest_pred_index]

            # print(f"Predictions: {predictions}")
            
            # print(f"Highest prediction index: {highest_pred_index}")
            # print(f"Predicted class: {predicted_class}")
            # print(response_data)
            return response_data['response']

        except requests.exceptions.RequestException as e:
            return JsonResponse({'Error in image classification function: ': f"{e}"})


def is_uptrend(data):
  image_class = image_classification(data=data)
#   print(f'Uptrend Function: {image_class}')
  if image_class == 'uptrend':
    return True
  else:
    return False


def is_downtrend(data):
  image_class = image_classification(data=data)
#   print(f'Downtrend Function: {image_class}')
  if image_class == 'downtrend':
    return True
  else:
    return False


def is_ranging_market(data):
  image_class = image_classification(data=data)
#   print(f'Ranging Market Function: {image_class}')
  if image_class == 'ranging market':
    return True
  else:
    return False


def support_and_resistance(df):
    peaks_range = [2, 3]
    num_peaks = -999

    sample_df = df
    sample = sample_df['Close'].to_numpy().flatten()
    sample_original = sample.copy()

    maxima = argrelextrema(sample, np.greater)
    minima = argrelextrema(sample, np.less)

    extrema = np.concatenate((maxima, minima), axis=1)[0]
    extrema_prices = np.concatenate((sample[maxima], sample[minima]))
    interval = extrema_prices[0] / 10000

    bandwidth = interval

    while num_peaks < peaks_range[0] or num_peaks > peaks_range[1]:
        initial_price = extrema_prices[0]
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(extrema_prices.reshape(-1, 1))

        a, b = min(extrema_prices), max(extrema_prices)
        price_range = np.linspace(a, b, 1000).reshape(-1, 1)

        pdf = np.exp(kde.score_samples(price_range))
        peaks = find_peaks(pdf)[0]
        num_peaks = len(peaks)
        bandwidth += interval

        if bandwidth > 100 * interval:
            print('Failed to converge, stopping...')
            break

    new_price_range = price_range[peaks]
    new_price_range = np.delete(new_price_range, 1, axis=0)

    # Return support and resistance levels
    return new_price_range


def is_support_level(data):
    support_level = support_and_resistance(data)[0][0]
    latest_price = data.iloc[-1].Close
    return latest_price <= support_level


def is_resistance_level(data):
    resistance_level = support_and_resistance(data)[1][0]
    latest_price = data.iloc[-1].Close
    return latest_price >= resistance_level


def is_asian_range_buy(asset):
    try:
        # Specify your local time zone
        local_timezone = pytz.timezone('Africa/Johannesburg')

        # Get the current time in the specified time zone
        current_time_local = datetime.now(local_timezone)

        # Extract the current hour
        current_hour_local = current_time_local.hour

        # Ensure that the algorithm runs after London Open in South African Time
        if current_hour_local >= 9:
            dataset = obtain_dataset(asset=asset, interval='1d', num_days=31)
            uptrend = is_uptrend(data=dataset)

            if uptrend:
                df = obtain_dataset(asset=asset, interval='1m', num_days=1)
                df.index = pd.to_datetime(df.index)
                today = datetime.now(local_timezone).strftime("%Y-%m-%d")
                df_today = df[df.index.strftime("%Y-%m-%d") == today]
                asian_range = df_today.between_time('01:00', '05:00')
                ranging_market = is_ranging_market(data=asian_range)

                if ranging_market:
                    levels = support_and_resistance(asian_range)
                    support_level = levels[0][0]
                    resistance_level = levels[1][0]
                    last_close = df.iloc[-1].Close

                    if uptrend and last_close >= resistance_level:
                        return True

            return False # no trade

        return False # no trade

    except Exception as e:
        print(f'Exception occured in asian_range_buy: {e} with asset: {asset}')
        return False # no trade


def is_asian_range_sell(asset):
    try:
        # Specify your local time zone
        local_timezone = pytz.timezone('Africa/Johannesburg')

        # Get the current time in the specified time zone
        current_time_local = datetime.now(local_timezone)

        # Extract the current hour
        current_hour_local = current_time_local.hour

        # Ensure that the algorithm runs after London Open in South African Time
        if current_hour_local >= 9:
            dataset = obtain_dataset(asset=asset, interval='1d', num_days=31)
            downtrend = is_downtrend(data=dataset)

            if downtrend:
                df = obtain_dataset(asset=asset, interval='1m', num_days=1)
                df.index = pd.to_datetime(df.index)
                today = datetime.now(local_timezone).strftime("%Y-%m-%d")
                df_today = df[df.index.strftime("%Y-%m-%d") == today]
                asian_range = df_today.between_time('01:00', '05:00')
                ranging_market = is_ranging_market(data=asian_range)

                if ranging_market:
                    levels = support_and_resistance(asian_range)
                    support_level = levels[0][0]
                    resistance_level = levels[1][0]
                    last_close = df.iloc[-1].Close
                    
                    if downtrend and last_close <= support_level:
                        return True


            return False # no trade

        return False # no trade

    except Exception as e:
        print(f'Exception occured in asian_range_sell: {e} with asset: {asset}')
        return False # no trade


def is_fibonacci_level(data, trend, level):
    # Ensure trend is either 'uptrend' or 'downtrend'
    try:
        trend = trend.lower()
        if trend not in ['uptrend', 'downtrend']:
            raise ValueError("Trend must be 'uptrend' or 'downtrend'")
        
        # Get the lowest close price and highest high price
        Low = data['Close'].min()
        High = data['High'].max()
        latest_price = data.iloc[-1]['Close']

        # Calculate the difference
        Diff = High - Low

        # Calculate Fibonacci levels based on the trend
        if trend == 'downtrend':
            Fib100 = High
            Fib618 = High - (Diff * 0.618)
            Fib50 = High - (Diff * 0.5)
            Fib382 = High - (Diff * 0.382)
            Fib236 = High - (Diff * 0.236)
            Fib0 = Low
        else:  # 'uptrend'
            Fib100 = Low
            Fib618 = Low + (Diff * 0.618)
            Fib50 = Low + (Diff * 0.5)
            Fib382 = Low + (Diff * 0.382)
            Fib236 = Low + (Diff * 0.236)
            Fib0 = High

        # Check if the latest price is below the specified Fibonacci level
        if trend == 'downtrend':
            if level == 0 and latest_price >= Fib0:
                return True
            elif level == 23.6 and latest_price >= Fib236:
                return True
            elif level == 38.2 and latest_price >= Fib382:
                return True
            elif level == 50 and latest_price >= Fib50:
                return True
            elif level == 61.8 and latest_price >= Fib618:
                return True
            elif level == 100 and latest_price >= Fib100:
                return True
        elif trend == 'uptrend':
            if level == 0 and latest_price <= Fib0:
                return True
            elif level == 23.6 and latest_price <= Fib236:
                return True
            elif level == 38.2 and latest_price <= Fib382:
                return True
            elif level == 50 and latest_price <= Fib50:
                return True
            elif level == 61.8 and latest_price <= Fib618:
                return True
            elif level == 100 and latest_price <= Fib100:
                return True
        
        return False
    except Exception as e:
        return JsonResponse({"error": f"Error occured in Fibonacci function: {e}"})


def is_ote_buy(asset):
    try:
        dataset = obtain_dataset(asset=asset, interval='1d', num_days=213)
        ranging_market = is_ranging_market(data=dataset)
        if ranging_market:
            latest_price = dataset.iloc[-1].Close
            support_level = support_and_resistance(dataset)[0][0]
            resistance_level = support_and_resistance(dataset)[1][0]
            if latest_price >= support_level and latest_price <= resistance_level:
                prev_4_days_data = obtain_dataset(asset=asset, interval='1h', num_days=4)
                uptrend = is_uptrend(data=prev_4_days_data)
                if uptrend:
                    prev_2_days_data = obtain_dataset(asset=asset, interval='1h', num_days=2)
                    if is_fibonacci_level(data=prev_2_days_data, trend='uptrend', level=50):
                        return True
    except Exception as e:
        print(f'Error occured in ote_buy_function: {e}')
        return JsonResponse({'message': f'Error occured in ote_buy_function: {e}'})


def is_ote_sell(asset):
    try:
        dataset = obtain_dataset(asset=asset, interval='1d', num_days=213)
        ranging_market = is_ranging_market(data=dataset)
        if ranging_market:
            latest_price = dataset.iloc[-1].Close
            support_level = support_and_resistance(dataset)[0][0]
            resistance_level = support_and_resistance(dataset)[1][0]
            if latest_price >= support_level and latest_price <= resistance_level:
                prev_4_days_data = obtain_dataset(asset=asset, interval='1h', num_days=4)
                downtrend = is_downtrend(data=prev_4_days_data)
                if downtrend:
                    prev_2_days_data = obtain_dataset(asset=asset, interval='1h', num_days=2)
                    if is_fibonacci_level(data=prev_2_days_data, trend='downtrend', level=50):
                        return True
    except Exception as e:
        print(f'Error occured in ote_sell_function: {e}')
        return JsonResponse({'message': f'Error occured in ote_sell_function: {e}'})


def is_bearish_orderblock(asset, tolerance=0.005, timeframe='1d'):
    try:
        dataset = obtain_dataset(asset=asset, interval=timeframe, num_days=214)
        ranging_market = is_ranging_market(data=dataset)
        
        if not ranging_market:
            return False
        
        levels = support_and_resistance(dataset)
        support_level = levels[0][0]
        resistance_level = levels[1][0]
        
        last_close = dataset.iloc[-1].Close
        
        if last_close <= support_level:
            return False
        
        # Loop through the dataset in reverse order to find the most recent candle where the resistance level is between the low and high
        for index, row in dataset[::-1].iterrows():
            if row['Low'] <= resistance_level <= row['High']:
                orderblock_open = row['High']
                orderblock_close = row['Low']
                
                # Check if the last_close is within the tolerance range of the orderblock_close
                if abs(last_close - orderblock_close) <= tolerance * orderblock_close and last_close < resistance_level:
                    return True    
        return False
    except Exception as e:
        print(f'Error occured in bearish orderblock function: {e}')
        return JsonResponse({'message': f'Error occured in bearish orderblock function: {e}'})


def is_bullish_orderblock(asset, tolerance=0.005, timeframe='1d'):
    try:
        dataset = obtain_dataset(asset=asset, interval=timeframe, num_days=214)
        ranging_market = is_ranging_market(data=dataset)
        
        if not ranging_market:
            return False
        
        levels = support_and_resistance(dataset)
        support_level = levels[0][0]
        resistance_level = levels[1][0]
        
        last_close = dataset.iloc[-1].Close
        
        if last_close >= resistance_level:
            return False
        
        # Loop through the dataset in reverse order to find the most recent candle where the resistance level is between the low and high
        for index, row in dataset[::-1].iterrows():
            if row['Low'] <= support_level <= row['High']:
                orderblock_open = row['High']
                orderblock_close = row['Low']
                
                # Check if the last_close is within the tolerance range of the orderblock_close
                if abs(last_close - orderblock_close) <= tolerance * orderblock_close and last_close > support_level:
                    return True
        return False
    except Exception as e:
        print(f'Error occured in bullish orderblock function: {e}')
        return JsonResponse({'message': f'Error occured in bullish orderblock function: {e}'})


def is_bullish_weekly_profile(asset):
    try:
        dataframe = obtain_dataset(asset=asset, interval='1wk', num_days=1)
        weekly_opening_price = dataframe.iloc[0]['Open']
        current_price = dataframe.iloc[-1]['Close']
        if current_price > weekly_opening_price:
            return True
        else:
            return False
    except Exception as e:
        print(f'Error occured in weekly bullish profile function: {e}')
        return JsonResponse({'message': f'Error occured in weekly bullish function: {e}'})


def is_bearish_weekly_profile(asset):
    try:
        dataframe = obtain_dataset(asset=asset, interval='1wk', num_days=1)
        weekly_opening_price = dataframe.iloc[0]['Open']
        current_price = dataframe.iloc[-1]['Close']
        if current_price < weekly_opening_price:
            return True
        else:
            return False
    except Exception as e:
        print(f'Error occured in weekly bearish profile function: {e}')
        return JsonResponse({'message': f'Error occured in weekly bearish function: {e}'})


@csrf_exempt
def genesys_live(request, identifier, num_positions, asset, interval, order_ticket, bot_id):
    
    return_statement = None
    percentage_test = 0

    asset = asset
    # print(f'Identifier: {identifier}\nInitial Equity: {initial_equity}\nTrade Equity: {trade_equity}\nNum Positions: {num_positions}\n')
    
    # def set_take_profit(number, type_of_setting):
    #     # current_equity = equity
    #     # print(f'Current Equity: {current_equity}\n')
    #     type_of_setting = type_of_setting.upper()
    #     number = float(number)
    #     nonlocal percentage_test, return_statement
            
    #     if type_of_setting == 'PERCENTAGE':
    #         percentage = ((current_equity - trade_equity) / initial_equity) * 100

    #         # variable = f'Identifier: {identifier}\nInitial Equity: {initial_equity}\nTrade Equity: {trade_equity}\nNum Positions: {num_positions}\nPercentage is: {percentage}\n'
    #         # percentage_test = variable
    #         if percentage >= number:
    #             return_statement = "close_position"
    #     elif type_of_setting == 'NUMBER':
    #         difference = current_equity - trade_equity
    #         if difference >= number:
    #             return_statement = "close_position"
        
    # def set_stop_loss(number, type_of_setting):
    #     nonlocal return_statement, percentage_test
    #     type_of_setting = type_of_setting.upper()
    #     number = -(float(number))
    #     # Get 'equity' here from the 'GenesysLive' model.
    #     if type_of_setting == 'PERCENTAGE':
    #         percentage = ((current_equity - trade_equity) / initial_equity) * 100
    #         percentage_test = percentage
    #         if percentage <= number:
    #             return_statement = "close_position"
    #     elif type_of_setting == 'NUMBER':
    #         difference = current_equity - trade_equity
    #         if difference <= number:
    #             return_statement = "close_position"
                    
    
    model_query = GenesysLive.objects.filter(model_id=identifier)

    if len(model_query) == 0:
        return JsonResponse({"message": f"Model has no such identifier"})
    
    # Check if there is any trade with the given model_id and order_ticket
    model_exists = uniqueBot.objects.filter(bot_id=bot_id).exists()

    # Return the appropriate response based on whether the model exists or not
    if model_exists:
        return JsonResponse({"message": "Model already has an ongoing position"})


    today = datetime.now().date()  # Get the current date
    model_traded = tradeModel.objects.filter(asset=asset, date_taken__date=today, model_id=identifier, timeframe=interval).exists()  # Filter by date only
    if model_traded:
        return JsonResponse({"message": f"Model Has Already Taken a trade for today"})


    if interval == '1d' or interval == '1wk':
        number_of_days = 213
    elif interval == '15m' or interval == '5m':
        number_of_days = 3
    else:
        number_of_days = 14

    dataset = obtain_dataset(asset=asset, interval=interval, num_days=number_of_days)
    
    model_code = model_query[0].model_code

    # test_model_id = 5505503

    # Inside genesys_live function
    try:
        # Execute model_code within a namespace dictionary
         # Initialize the namespace dictionary with functions
        namespace = {
            # 'set_take_profit': set_take_profit,
            # 'set_stop_loss': set_stop_loss,
            'interval': interval,
            'num_positions': num_positions,
            'is_support_level': is_support_level,
            'is_resistance_level': is_resistance_level,
            'dataset': dataset,
            'is_uptrend': is_uptrend,
            'is_downtrend': is_downtrend,
            'is_ranging_market': is_ranging_market,
            'is_bullish_candle': is_bullish_candle,
            'is_bearish_candle': is_bearish_candle,
            'is_bullish_engulfing': is_bullish_engulfing,
            'is_bearish_engulfing': is_bearish_engulfing,
            'is_morning_star': is_morning_star,
            'is_evening_star': is_evening_star,
            'is_three_white_soldiers': is_three_white_soldiers,
            'is_three_black_crows': is_three_black_crows,
            'is_morning_doji_star': is_morning_doji_star,
            'is_evening_doji_star': is_evening_doji_star,
            'is_rising_three_methods': is_rising_three_methods,
            'is_falling_three_methods': is_falling_three_methods,
            'is_hammer': is_hammer,
            'is_hanging_man': is_hanging_man,
            'is_inverted_hammer': is_inverted_hammer,
            'is_shooting_star': is_shooting_star,
            'is_bullish_kicker': is_bullish_kicker,
            'is_bearish_kicker': is_bearish_kicker,
            'is_bullish_harami': is_bullish_harami,
            'is_bearish_harami': is_bearish_harami,
            'is_bullish_three_line_strike': is_bullish_three_line_strike,
            'is_bearish_three_line_strike': is_bearish_three_line_strike,
            'moving_average': moving_average,
            'bbands': bbands,
            'momentum': momentum,
            'rsi': rsi,
            'is_asian_range_buy': is_asian_range_buy,
            'is_asian_range_sell': is_asian_range_sell,
            'asset': asset,
            'is_fibonacci_level': is_fibonacci_level,
            'is_ote_buy': is_ote_buy,
            'is_ote_sell': is_ote_sell,
            'is_bullish_orderblock': is_bullish_orderblock,
            'is_bearish_orderblock': is_bearish_orderblock,
            'is_bullish_weekly_profile': is_bullish_weekly_profile,
            'is_bearish_weekly_profile': is_bearish_weekly_profile
        }
    
        exec(model_code, namespace)
        
        # Retrieve return_statement from the namespace
        if return_statement == None:
            return_statement = namespace.get('return_statement', None)
    except Exception as e:
        # Log and handle exceptions
        print(f"Error executing model code: {e} with interval: {interval}\n with dataset:\n {dataset} with length of {len(dataset)}")
        return JsonResponse({"error": f"{e} with interval: {interval}"})

    # Check return_statement and handle accordingly
    if return_statement:
        return JsonResponse({"message": return_statement})
    else:
        return JsonResponse({"message": f"No message to send order from backend"})


@csrf_exempt
def delete_unique_bot(request, bot_id):
    try:
        bot = uniqueBot.objects.filter(bot_id=bot_id)
        bot.delete()
        return JsonResponse({"message": f"Bot deleted Successfully!"})
    except Exception as e:
        return JsonResponse({"message": f"Error occured in Delete Bot Function: {e}"})


@csrf_exempt
def clear_stuff(request):
    uniqueBot.objects.all().delete()
    # tradeModel.objects.all().delete()
    # Trade.objects.all().delete()
    return JsonResponse({"message": f"All models deleted!"})


@csrf_exempt
def save_genesys_model(request):
    
    if request.method == 'POST':
        model_id = ''
        model_code = ''
        true_initial_equity = ''
        try:
            data = json.loads(request.body)
            model_id = int(data.get('model_id'))
            model_code = data.get('model_code')
            true_initial_equity = float(data.get('true_initial_equity'))
            
            # Save the data to your model
            new_model = GenesysLive(
                model_id=model_id,
                model_code=model_code,
                true_initial_equity=true_initial_equity
            )
            new_model.save()
            
            return JsonResponse({'message': f'Model saved successfully\n model_id: {model_id}, true_initial_equity: {true_initial_equity}'})
        except Exception as e:
            return JsonResponse({'message': f'{e}\nmodel_id: {model_id}, true_initial_equity: {true_initial_equity}'})
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)


@csrf_exempt
def test_date(request, asset):
    today = datetime.now().date()  # Get the current date
    model_traded = tradeModel.objects.filter(asset=asset, date_taken__date=today).exists()  # Filter by date only
    if model_traded:
        return JsonResponse({"message": f"Model Has Already Taken a trade for today: {model_traded}"})
    else:
        return JsonResponse({"message": f"Model Has Taken No trade for the day: {model_traded}"})



@csrf_exempt
def save_new_trade_model(request, model_id, initial_equity, order_ticket, asset, volume, type_of_trade, timeframe, bot_id):
    try:
        model_query = GenesysLive.objects.filter(model_id=model_id)
        if len(model_query) == 0:
            return JsonResponse({"message": f"Model has no such identifier"})
        model_code = model_query[0].model_code
        today = datetime.now()  # Get the current date and time
        new_trade_model = tradeModel(model_id=model_id, model_code=model_code, initial_equity=initial_equity, order_ticket=order_ticket, type_of_trade=type_of_trade, volume=volume, asset=asset, profit=-1.0, timeframe=timeframe, date_taken=today)
        new_trade_model.save()
        unique_bot = uniqueBot(model_id=model_id, order_ticket=order_ticket, asset=asset, bot_id=bot_id)
        unique_bot.save()
        return JsonResponse({'message': f'Saved New Model Successfully!'})
    except Exception as e:
        return JsonResponse({'message': f'Error Occured in Save New Trade Model Function: {e}'})


@csrf_exempt
def update_trade_model(request, model_id, order_ticket, profit):
    
    try:
        profit = float(profit)
        model_query = tradeModel.objects.filter(model_id=model_id, order_ticket=order_ticket)
        if len(model_query) == 0:
            return JsonResponse({"message": f"Model has no such identifier"})
        # Update Profit if Match is Found
        model_query.update(profit=profit)
        return JsonResponse({'message': 'Model Updated Successfully!'})
    except Exception as e:
        return JsonResponse({'message': f'Exception Occured In Update Trade Model Function: {e}'})


@csrf_exempt
def get_model_performance(request):
    if request.method == 'GET':
        models = tradeModel.objects.values(
            'model_id', 
            'model_code', 
            'initial_equity', 
            'order_ticket', 
            'asset', 
            'profit', 
            'volume', 
            'type_of_trade', 
            'timeframe', 
            'date_taken'
        )

        # Remove duplicates by using a dictionary to track unique order_tickets
        unique_models = {}
        for model in models:
            if model['order_ticket'] not in unique_models:
                unique_models[model['order_ticket']] = model
        
        data = list(unique_models.values())
        
        return JsonResponse(data, safe=False)
    else:
        return JsonResponse({'error': 'Invalid HTTP method'}, status=405)


@csrf_exempt
def get_user_assets(request, email='butterrobot83@gmail.com'):
    try:
        # Fetch unique asset names from tradeModel
        user_assets = tradeModel.objects.values_list('asset', flat=True).distinct()
        
        # Convert QuerySet to a list
        unique_assets = list(user_assets)

        return JsonResponse({'message': unique_assets})
    except Exception as e:
        print(f'Error occurred in get_user_assets: {e}')
        return JsonResponse({'error': f'Error occurred in get_user_assets: {e}'})



# @csrf_exempt
# def fetch_asset_data(request, asset):
#     try:
#         asset = asset.upper()
#         email = 'butterrobot83@gmail.com'
#         trade_data = Trade.objects.filter(email=email, asset=asset)

#         # Initialize variables
#         profit_list = []
#         win_count = 0
#         loss_count = 0
#         overall_return = 0

#         # Iterate over trade data to calculate the required values
#         for trade in trade_data:
#             amount = trade.amount  # Assuming 'amount' is the profit/loss attribute in Trade model
#             profit_list.append(amount)
#             overall_return += amount
#             if amount >= 0:
#                 win_count += 1
#             elif amount < 0:
#                 loss_count += 1

#         total_trades = len(trade_data)
#         win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0
#         loss_rate = 100 - win_rate

#         return JsonResponse({
#             'profit_list': profit_list,
#             'win_rate': win_rate,
#             'loss_rate': loss_rate,
#             'overall_return': overall_return
#         })

#     except Exception as e:
#         print(f'Error occurred in fetch_asset_data: {e}')
#         return JsonResponse({'error': f'Error occurred in fetch_asset_data: {e}'})


@csrf_exempt
def fetch_asset_data(request, asset):
    try:
        asset = asset.upper()
        email = 'butterrobot83@gmail.com'
        trade_data = Trade.objects.filter(email=email, asset=asset)

        # Initialize variables for overall stats
        profit_list = []
        win_count = 0
        loss_count = 0
        overall_return = 0

        # Initialize dictionaries to store strategy-based metrics
        strategy_profit_list = {}
        strategy_win_count = {}
        strategy_loss_count = {}
        strategy_overall_return = {}

        # Iterate over trade data to calculate the required values
        for trade in trade_data:
            amount = trade.amount  # Assuming 'amount' is the profit/loss attribute in Trade model
            profit_list.append(amount)
            overall_return += amount

            if amount >= 0:
                win_count += 1
            else:
                loss_count += 1

            strategy = trade.strategy
            if strategy not in strategy_profit_list:
                strategy_profit_list[strategy] = []
                strategy_win_count[strategy] = 0
                strategy_loss_count[strategy] = 0
                strategy_overall_return[strategy] = 0

            strategy_profit_list[strategy].append(amount)
            strategy_overall_return[strategy] += amount

            if amount >= 0:
                strategy_win_count[strategy] += 1
            else:
                strategy_loss_count[strategy] += 1

        # Calculate overall win rate and loss rate
        total_trades = len(trade_data)
        win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0
        loss_rate = 100 - win_rate

        # Calculate win rate, loss rate, and overall return for each strategy
        strategy_metrics = {}
        for strategy in strategy_profit_list:
            total_trades_strategy = len(strategy_profit_list[strategy])
            strategy_win_rate = (strategy_win_count[strategy] / total_trades_strategy) * 100 if total_trades_strategy > 0 else 0
            strategy_loss_rate = 100 - strategy_win_rate

            strategy_metrics[strategy] = {
                'profit_list': strategy_profit_list[strategy],
                'win_rate': strategy_win_rate,
                'loss_rate': strategy_loss_rate,
                'overall_return': strategy_overall_return[strategy]
            }

        return JsonResponse({
            'overall': {
                'profit_list': profit_list,
                'win_rate': win_rate,
                'loss_rate': loss_rate,
                'overall_return': overall_return
            },
            'strategy_metrics': strategy_metrics
        })

    except Exception as e:
        print(f'Error occurred in fetch_asset_data: {e}')
        return JsonResponse({'error': f'Error occurred in fetch_asset_data: {e}'})


@csrf_exempt
def fetch_asset_data_from_models(request, asset):
    try:
        asset = asset.upper()
        model_data = tradeModel.objects.filter(asset=asset)
        
        # Get distinct model IDs for the given asset
        model_ids = model_data.values_list('model_id', flat=True).distinct()
        response_data = []

        for model_id in model_ids:
            model_trades = model_data.filter(model_id=model_id)
            profits = [trade.profit for trade in model_trades]
            equity_curve = calculate_equity_curve(profits)
            win_rate, loss_rate, overall_return = calculate_performance_metrics(profits)
            
            response_data.append({
                'model_id': model_id,
                'equity_curve': equity_curve,
                'win_rate': win_rate,
                'loss_rate': loss_rate,
                'overall_return': overall_return,
            })

        return JsonResponse({'data': response_data})

    except Exception as e:
        print(f'Error occurred in fetch_asset_data_from_models: {e}')
        return JsonResponse({'error': f'Error occurred in fetch_asset_data_from_models: {e}'})


def calculate_equity_curve(profits):
    initial_equity = 10000
    equity = initial_equity
    equity_curve = [equity]
    for profit in profits:
        equity += profit
        equity_curve.append(equity)
    return equity_curve


def calculate_performance_metrics(profits):
    total_trades = len(profits)
    winning_trades = len([profit for profit in profits if profit > 0])
    losing_trades = len([profit for profit in profits if profit < 0])
    
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    loss_rate = (losing_trades / total_trades) * 100 if total_trades > 0 else 0
    overall_return = sum(profits)
    
    return win_rate, loss_rate, overall_return


@csrf_exempt
def get_asset_summary(request, asset):
    try:
        asset = asset.upper()
        asset_news_summary = dailyBrief.objects.filter(asset=asset)[0].summary
        return JsonResponse({'message': f'{asset_news_summary}'})
    except Exception as e:
        print(f'Error occurred in get_asset_daily_brief_data: {e}')
        return JsonResponse({'error': f'Error occurred in get_asset_daily_brief_data: {e}'})


@csrf_exempt
def generate_cot_data(request):
    try:
        # Get requested assets from POST data if provided
        if request.method == 'POST':
            requested_assets = json.loads(request.body).get('assets', [])
        else:
            # Default assets for GET requests
            requested_assets = [
                'USD INDEX - ICE FUTURES U.S.',
                'EURO FX - CHICAGO MERCANTILE EXCHANGE',
                'BRITISH POUND - CHICAGO MERCANTILE EXCHANGE',
                'CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE',
                'SWISS FRANC - CHICAGO MERCANTILE EXCHANGE',
                'JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE',
                'NZ DOLLAR - CHICAGO MERCANTILE EXCHANGE',
                'AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE',
                'GOLD - COMMODITY EXCHANGE INC.',
                'UST BOND - CHICAGO BOARD OF TRADE',
                'UST 10Y NOTE - CHICAGO BOARD OF TRADE',
                'UST 5Y NOTE - CHICAGO BOARD OF TRADE',
                'NASDAQ MINI - CHICAGO MERCANTILE EXCHANGE',
                'E-MINI S&P 500 -',
                'DOW JONES U.S. REAL ESTATE IDX - CHICAGO BOARD OF TRADE'
            ]

        # Get the current year and previous year
        current_year = pd.Timestamp.now().year
        previous_year = current_year - 1

        # Create list to store DataFrames
        df_list = []

        # Fetch data for previous and current year
        for year in range(previous_year, current_year + 1):
            single_year = cot.cot_year(year, cot_report_type='legacy_futopt')
            df_list.append(single_year)

        # Concatenate all DataFrames
        df = pd.concat(df_list, ignore_index=True)

        # Convert dates to datetime
        df['As of Date in Form YYYY-MM-DD'] = pd.to_datetime(df['As of Date in Form YYYY-MM-DD'])

        # Filter for current year data
        unfiltered_currency_df = df[df['As of Date in Form YYYY-MM-DD'].dt.year == current_year]

        # Filter for requested assets
        unfiltered_currency_df = unfiltered_currency_df[
            unfiltered_currency_df['Market and Exchange Names'].isin(requested_assets)
        ]

        # Remove specific exclusions (e.g., MICRO GOLD)
        unfiltered_currency_df = unfiltered_currency_df[
            unfiltered_currency_df['Market and Exchange Names'] != 'MICRO GOLD - COMMODITY EXCHANGE INC.'
        ]

        # Fill missing values and ensure numeric columns
        numeric_columns = [
            'Noncommercial Positions-Long (All)',
            'Noncommercial Positions-Short (All)',
            'Commercial Positions-Long (All)',
            'Commercial Positions-Short (All)'
        ]
        
        unfiltered_currency_df[numeric_columns] = unfiltered_currency_df[numeric_columns].fillna(0).astype(float)

        # Calculate net positions for unfiltered data
        unfiltered_currency_df['Net Noncommercial Positions'] = (
            unfiltered_currency_df['Noncommercial Positions-Long (All)'] - 
            unfiltered_currency_df['Noncommercial Positions-Short (All)']
        )
        unfiltered_currency_df['Net Commercial Positions'] = (
            unfiltered_currency_df['Commercial Positions-Long (All)'] - 
            unfiltered_currency_df['Commercial Positions-Short (All)']
        )

        # Get the rows with maximum open interest for each market
        idx = unfiltered_currency_df.groupby('Market and Exchange Names')['Open Interest (All)'].idxmax()
        currency_df = unfiltered_currency_df.loc[idx]

        # Calculate total positions
        currency_df['Total Noncommercial Positions'] = (
            currency_df['Noncommercial Positions-Long (All)'] + 
            currency_df['Noncommercial Positions-Short (All)']
        )
        currency_df['Total Commercial Positions'] = (
            currency_df['Commercial Positions-Long (All)'] + 
            currency_df['Commercial Positions-Short (All)']
        )
        currency_df['Total Positions'] = (
            currency_df['Total Noncommercial Positions'] + 
            currency_df['Total Commercial Positions']
        )

        # Calculate percentages
        currency_df['Percentage Noncommercial Long'] = (
            currency_df['Noncommercial Positions-Long (All)'] / 
            currency_df['Total Noncommercial Positions']
        ) * 100
        currency_df['Percentage Noncommercial Short'] = (
            currency_df['Noncommercial Positions-Short (All)'] / 
            currency_df['Total Noncommercial Positions']
        ) * 100
        currency_df['Percentage Commercial Long'] = (
            currency_df['Commercial Positions-Long (All)'] / 
            currency_df['Total Commercial Positions']
        ) * 100
        currency_df['Percentage Commercial Short'] = (
            currency_df['Commercial Positions-Short (All)'] / 
            currency_df['Total Commercial Positions']
        ) * 100

        # Generate plots
        plot_urls = plot_net_positions(unfiltered_currency_df)  # Assuming this function exists

        # Prepare response data
        data = {}
        round_off_number = 2

        for asset in requested_assets:
            asset_df = currency_df[currency_df['Market and Exchange Names'] == asset]
            
            if not asset_df.empty:
                latest_data = asset_df.iloc[0]
                data[asset] = {
                    'Date': latest_data['As of Date in Form YYYY-MM-DD'].strftime('%Y-%m-%d'),
                    'Percentage Noncommercial Long': round(latest_data['Percentage Noncommercial Long'], round_off_number),
                    'Percentage Noncommercial Short': round(latest_data['Percentage Noncommercial Short'], round_off_number),
                    'Percentage Commercial Long': round(latest_data['Percentage Commercial Long'], round_off_number),
                    'Percentage Commercial Short': round(latest_data['Percentage Commercial Short'], round_off_number),
                    'Plot URL': plot_urls.get(asset, '')
                }

        return JsonResponse(data)
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def plot_net_positions(df):
    # Get the unique currencies
    unique_currencies = df['Market and Exchange Names'].unique()
    plot_urls = {}

    # Set the plot style
    sns.set(style="whitegrid")

    for currency in unique_currencies:
        # Filter DataFrame for the current currency
        currency_data = df[df['Market and Exchange Names'] == currency]

        # Calculate min, max, and midpoint for Net Commercial Positions
        min_commercial = currency_data['Net Commercial Positions'].min()
        max_commercial = currency_data['Net Commercial Positions'].max()
        midpoint_commercial = (min_commercial + max_commercial) / 2

        # Create a figure with two subplots: one for net positions and one for Open Interest
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Plot net noncommercial and net commercial positions on the first subplot (ax1)
        ax1.plot(currency_data['As of Date in Form YYYY-MM-DD'], 
                 currency_data['Net Noncommercial Positions'], 
                 label='Net Noncommercial Positions', color='blue')
        ax1.plot(currency_data['As of Date in Form YYYY-MM-DD'], 
                 currency_data['Net Commercial Positions'], 
                 label='Net Commercial Positions', color='red')
        ax1.axhline(y=midpoint_commercial, color='green', linestyle='--', 
                    label=f'Midpoint of Net Commercial Positions ({midpoint_commercial:.2f})')

        # Customize the first subplot (Net Positions)
        ax1.set_ylabel('Net Positions')
        ax1.legend(loc='upper left')
        ax1.set_title(f'Net Positions and Open Interest for {currency}')
        ax1.grid(True)

        # Plot Open Interest on the second subplot (ax2)
        ax2.plot(currency_data['As of Date in Form YYYY-MM-DD'], 
                 currency_data['Open Interest (All)'], 
                 label='Open Interest', color='purple', linestyle='-')

        # Customize the second subplot (Open Interest)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Open Interest')
        ax2.legend(loc='upper left')
        ax2.grid(True)

        # Rotate the x-axis labels for better readability
        for label in ax2.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the plot to a bytes buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_url = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plot_urls[currency] = f'data:image/png;base64,{plot_url}'

        plt.close()

    return plot_urls


@csrf_exempt
def create_chill_data(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            section = data.get('section')
            text = data.get('text')
            
            # Check if a Chill instance with the same section already exists
            if Chill.objects.filter(section=section).exists():
                return JsonResponse({'message': 'Chill Data with this section already exists!'}, status=400)
            
            # If not, save the new data
            chill_data = Chill(
                section=section,
                text=text
            )
            chill_data.save()
            return JsonResponse({'message': 'Chill Data Saved Successfully!'}, status=201)
        
        except Exception as e:
            print(f'Error in Chill Data Function: {e}')
            return JsonResponse({'message': f'Error in Chill Data Function: {e}'}, status=500)
    else:
        return JsonResponse({'message': 'Invalid request method.'}, status=405)


@csrf_exempt
def fetch_chill_sections(request):
    try:
        sections = Chill.objects.all().order_by('id')  # Sort by id to maintain order
        section_list = [{'section': section.section, 'text': section.text} for section in sections]
        return JsonResponse({'sections': section_list}, status=200)
    except Exception as e:
        return JsonResponse({'message': str(e)}, status=500)


@csrf_exempt
def fetch_chill_data(request):
    try:
        if request.method == 'GET':
            section = request.GET.get('section')
            if section:
                chill_data = Chill.objects.filter(section=section).first()  # Fetch the Chill data based on the section
                if chill_data:
                    return JsonResponse({'section': chill_data.section, 'text': chill_data.text})
                else:
                    return JsonResponse({'message': 'Section not found'}, status=404)
            else:
                return JsonResponse({'message': 'No section provided'}, status=400)
        else:
            return JsonResponse({'message': 'Invalid request method'}, status=405)
    except Exception as e:
        return JsonResponse({'message': f'Error fetching Chill data: {e}'})


@csrf_exempt
def edit_chill_data(request):
    try:
        if request.method == 'POST':
            body = json.loads(request.body)
            section = body.get('section')
            text = body.get('text')

            if not section or not text:
                return JsonResponse({'message': 'Invalid data'}, status=400)

            # Use update() to update the record
            updated_count = Chill.objects.filter(section=section).update(text=text)

            if updated_count == 0:
                return JsonResponse({'message': 'Section not found'}, status=404)

            return JsonResponse({'message': 'Section updated successfully'}, status=200)
        else:
            return JsonResponse({'message': 'Invalid request method'}, status=405)
    except Chill.DoesNotExist:
        return JsonResponse({'message': 'Section not found'}, status=404)
    except Exception as e:
        return JsonResponse({'message': str(e)}, status=500)

@csrf_exempt
def delete_chill_entry(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            section = data.get('section')
            if section:
                Chill.objects.filter(section=section).delete()  # Deletes entry with the specified section name
                return JsonResponse({'message': 'Entry deleted successfully'}, status=200)
            return JsonResponse({'message': 'Section name not provided'}, status=400)
        except Chill.DoesNotExist:
            return JsonResponse({'message': 'Entry does not exist'}, status=404)
        except Exception as e:
            return JsonResponse({'message': str(e)}, status=500)
    return JsonResponse({'message': 'Invalid request method'}, status=405)


@csrf_exempt
def fetch_trading_images(request):
    try:
        # Define the base directory for images
        base_dir = os.path.join(os.path.dirname(__file__), 'image_folder')
        
        # Initialize an empty dictionary to hold folders and their images
        images_data = {}
        
        # Loop through each subfolder in the base directory
        for folder in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, folder)
            if os.path.isdir(folder_path):
                encoded_images = []
                # Read and encode each image file in Base64
                for img_file in os.listdir(folder_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                        img_path = os.path.join(folder_path, img_file)
                        with open(img_path, "rb") as image_file:
                            # Encode the image and add it to the list
                            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                            encoded_images.append({
                                "filename": img_file,
                                "data": f"data:image/{img_file.split('.')[-1]};base64,{encoded_string}"
                            })
                images_data[folder] = encoded_images
        
        return JsonResponse({"folders": images_data}, status=200)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def alert_bot(request):
    if request.method == "GET":
        # Fetch all alerts
        alerts = AlertBot.objects.all()
        alerts_data = [
            {
                "id": alert.id,
                "asset": alert.asset,
                "price": alert.price,
                "condition": alert.condition,
                "checked": alert.checked
            }
            for alert in alerts
        ]
        return JsonResponse({"alerts": alerts_data}, status=200)

    elif request.method == "POST":
        try:
            data = json.loads(request.body)
            
            if not isinstance(data, list):
                return JsonResponse({"error": "Expected a list of alerts."}, status=400)
            
            responses = []
            for alert in data:
                asset = alert.get("asset")
                price = alert.get("price")
                condition = alert.get("condition")

                if not asset or not price or not condition:
                    responses.append({"asset": asset, "error": "Missing required fields."})
                    continue

                # Ensure uniqueness per asset
                existing_alert = AlertBot.objects.filter(asset=asset).first()

                if existing_alert:
                    existing_alert.price = price
                    existing_alert.condition = condition
                    existing_alert.checked = False
                    existing_alert.save()
                    responses.append({"asset": asset, "message": "Asset alert updated."})
                else:
                    AlertBot.objects.create(asset=asset, price=price, condition=condition, checked=False)
                    responses.append({"asset": asset, "message": "Asset alert created."})
            
            return JsonResponse({"results": responses}, status=200)
        
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    elif request.method == "DELETE":
        try:
            alert_id = request.GET.get('id')
            if not alert_id:
                return JsonResponse({"error": "Alert ID is required."}, status=400)
            
            alert = AlertBot.objects.filter(id=alert_id).first()
            if not alert:
                return JsonResponse({"error": "Alert not found."}, status=404)
            
            alert.delete()
            return JsonResponse({"message": "Alert deleted successfully."}, status=200)
        
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
            
    else:
        return JsonResponse({"error": "Invalid request method."}, status=405)


def send_whatsapp_message(asset, message):
    
    # Twilio setup after fix
    ACCOUNT_SID = os.environ['TWILIO_SID']
    AUTH_TOKEN = os.environ['TWILIO_AUTH_TOKEN']
    TWILIO_CLIENT = Client(ACCOUNT_SID, AUTH_TOKEN)

    """
    Sends a WhatsApp message using Twilio.
    """
    
    TWILIO_CLIENT.messages.create(
        body=message,
        from_='whatsapp:+14155238886',
        to='whatsapp:+27847316417'
    )
    print(f"{asset}: {message}")



def manage_alerts():
    """
    Checks all alerts in the AlertBot model and sends notifications if conditions are met.
    Designed to be called periodically by an external scheduler.
    """

    alerts = AlertBot.objects.filter(checked=False)  # Fetch unchecked alerts
    for alert in alerts:
        try:
            asset = alert.asset
            target_price = alert.price
            condition = alert.condition  # e.g., "greater_than" or "less_than"

            # Fetch real-time data
            data = obtain_dataset(asset, interval="1m", num_days=1)

            if data.empty:
                print(f"No data available for {asset}. Skipping...")
                continue

            # Get the most recent closing price (ensure it's a scalar value)
            latest_price = round(float(data["Close"].iloc[-1]), 5)

            # Check the condition (compare scalar values only)
            condition_met = (
                (condition == ">" and latest_price > float(target_price)) or
                (condition == "<" and latest_price < float(target_price))
            )



            if condition_met:
                # Send notification
                message = (
                    f"Alert triggered for {asset}! Current price: {latest_price} "
                    f"{condition} {target_price}"
                )
                send_whatsapp_message(asset, message)

                # Mark the alert as checked
                alert.checked = True
                alert.save()

        except Exception as e:
            print(f"Error processing alert for {alert.asset}: {e}")


# Schedule the alert_bot function to run every 5 minutes
scheduler.add_job(
    manage_alerts,  # Replace with the name of your alert-checking function
    trigger=IntervalTrigger(minutes=5),
    id='manage_alerts_job',
    name='Check alerts every 5 minutes',
    replace_existing=True
)


@csrf_exempt
def create_finetuning_data(request):
    try:
        # Query all CHILL entries
        chill_data = Chill.objects.all()

        # Prepare dataset for JSONL format
        data_list = []
        for entry in chill_data:
            data_list.append({
                "messages": [
                    {"role": "system", "content": "TraderGPT is a trading assistant that provides advanced market analysis and trading strategies."},
                    {"role": "user", "content": entry.section},
                    {"role": "assistant", "content": entry.text}
                ]
            })

        # Define file path
        file_path = 'chill_data.jsonl'

        # Save as JSONL file
        with open(file_path, 'w') as jsonl_file:
            for item in data_list:
                jsonl_file.write(json.dumps(item) + '\n')

        # Serve the file as a download
        with open(file_path, 'rb') as jsonl_file:
            response = HttpResponse(jsonl_file.read(), content_type='application/jsonl')
            response['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
            return response
    except Exception as e:
        return JsonResponse({'message': f"Error occurred: \n{e}"})

@csrf_exempt
def create_image_finetuning_data(request):
    try:
        # Define the base directory for images
        base_dir = os.path.join(os.path.dirname(__file__), 'image_folder')

        # Prepare dataset for JSONL format
        data_list = []
        for root, _, files in os.walk(base_dir):
            for img_file in files:
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    img_path = os.path.join(root, img_file)
                    with open(img_path, "rb") as image_file:
                        # Encode image in Base64
                        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                        
                        # Create a single message for the image with proper formatting
                        image_content = {
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "TraderGPT is a trading assistant that provides advanced market analysis and trading strategies."
                                },
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": "What do you see in this image?"
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/{img_file.split('.')[-1]};base64,{encoded_string}"
                                            }
                                        }
                                    ]
                                },
                                {
                                    "role": "assistant",
                                    "content": "This is a trading chart. I can help analyze it."
                                }
                            ]
                        }
                        data_list.append(image_content)

        # Define file path
        file_path = 'image_data.jsonl'

        # Save as JSONL file
        with open(file_path, 'w') as jsonl_file:
            for item in data_list:
                jsonl_file.write(json.dumps(item) + '\n')

        # Serve the file as a download
        with open(file_path, 'rb') as jsonl_file:
            response = HttpResponse(jsonl_file.read(), content_type='application/jsonl')
            response['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
            return response

    except Exception as e:
        return JsonResponse({'message': f"Error occurred: {str(e)}"})


# @csrf_exempt
# def create_combined_finetuning_data(request):
#     try:
#         # Prepare combined dataset
#         data_list = []

#         # Step 1: Add CHILL data
#         chill_data = Chill.objects.all()
#         for entry in chill_data:
#             data_list.append({
#                 "messages": [
#                     {
#                         "role": "system",
#                         "content": "TraderGPT is a trading assistant that provides advanced market analysis and trading strategies."
#                     },
#                     {
#                         "role": "user",
#                         "content": entry.section
#                     },
#                     {
#                         "role": "assistant",
#                         "content": entry.text
#                     }
#                 ]
#             })

#         # Step 2: Add Image data
#         base_dir = os.path.join(os.path.dirname(__file__), 'image_folder')
#         for root, _, files in os.walk(base_dir):
#             for img_file in files:
#                 if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
#                     img_path = os.path.join(root, img_file)
#                     with open(img_path, "rb") as image_file:
#                         # Encode image in Base64
#                         encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                        
#                         # Create properly formatted image entry
#                         image_content = {
#                             "messages": [
#                                 {
#                                     "role": "system",
#                                     "content": "TraderGPT is a trading assistant that provides advanced market analysis and trading strategies."
#                                 },
#                                 {
#                                     "role": "user",
#                                     "content": [
#                                         {
#                                             "type": "text",
#                                             "text": "What do you see in this image?"
#                                         },
#                                         {
#                                             "type": "image_url",
#                                             "image_url": {
#                                                 "url": f"data:image/{img_file.split('.')[-1]};base64,{encoded_string}"
#                                             }
#                                         }
#                                     ]
#                                 },
#                                 {
#                                     "role": "assistant",
#                                     "content": "This is a trading chart. I can help analyze it."
#                                 }
#                             ]
#                         }
#                         data_list.append(image_content)

#         # Step 3: Save combined data as JSONL
#         file_path = 'combined_finetuning_data.jsonl'
#         with open(file_path, 'w') as jsonl_file:
#             for item in data_list:
#                 jsonl_file.write(json.dumps(item) + '\n')

#         # Step 4: Serve the file as a download
#         with open(file_path, 'rb') as jsonl_file:
#             response = HttpResponse(jsonl_file.read(), content_type='application/jsonl')
#             response['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
#             return response

#     except Exception as e:
#         return JsonResponse({'message': f"Error occurred: {str(e)}"})
        

@csrf_exempt
def create_combined_finetuning_data(request):
    try:
        # Settings for chunking
        MAX_ENTRIES_PER_FILE = 1000  # Adjust this number based on your needs
        
        # Prepare combined dataset
        all_data = []
        
        # Step 1: Add CHILL data
        chill_data = Chill.objects.all()
        for entry in chill_data:
            all_data.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "TraderGPT is a trading assistant that provides advanced market analysis and trading strategies."
                    },
                    {
                        "role": "user",
                        "content": entry.section
                    },
                    {
                        "role": "assistant",
                        "content": entry.text
                    }
                ]
            })

        # Step 2: Add Image data
        base_dir = os.path.join(os.path.dirname(__file__), 'image_folder')
        for root, _, files in os.walk(base_dir):
            for img_file in files:
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    img_path = os.path.join(root, img_file)
                    with open(img_path, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                        image_content = {
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "TraderGPT is a trading assistant that provides advanced market analysis and trading strategies."
                                },
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": "What do you see in this image?"
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/{img_file.split('.')[-1]};base64,{encoded_string}"
                                            }
                                        }
                                    ]
                                },
                                {
                                    "role": "assistant",
                                    "content": "This is a trading chart. I can help analyze it."
                                }
                            ]
                        }
                        all_data.append(image_content)

        # Step 3: Create chunks and save as separate JSONL files
        chunk_files = []
        for i in range(0, len(all_data), MAX_ENTRIES_PER_FILE):
            chunk = all_data[i:i + MAX_ENTRIES_PER_FILE]
            file_path = f'finetuning_data_part_{i//MAX_ENTRIES_PER_FILE + 1}.jsonl'
            chunk_files.append(file_path)
            
            with open(file_path, 'w') as jsonl_file:
                for item in chunk:
                    jsonl_file.write(json.dumps(item) + '\n')

        # Step 4: Create ZIP file containing all chunks
        zip_file_path = 'chunked_finetuning_data.zip'
        with zipfile.ZipFile(zip_file_path, 'w') as zipf:
            for file_path in chunk_files:
                zipf.write(file_path)
                
        # Step 5: Serve the ZIP file
        with open(zip_file_path, 'rb') as zip_file:
            response = HttpResponse(zip_file.read(), content_type='application/zip')
            response['Content-Disposition'] = f'attachment; filename="{os.path.basename(zip_file_path)}"'

        # Clean up temporary files
        for file_path in chunk_files:
            os.remove(file_path)
        os.remove(zip_file_path)

        return response

    except Exception as e:
        return JsonResponse({'message': f"Error occurred: {str(e)}"})


# Fetch all accounts
@csrf_exempt
def get_accounts(request):
    accounts = Account.objects.all()
    accounts_data = [
        {
            "id": account.id,
            "name": account.account_name,
            "initial_capital": account.initial_capital,
            "main_assets": account.main_assets,  # Include main_assets in the response
        }
        for account in accounts
    ]
    return JsonResponse(accounts_data, safe=False)


@csrf_exempt
def create_account(request):
    if request.method == "POST":
        data = json.loads(request.body)
        account_name = data.get("name")
        initial_capital = data.get("initial_capital")
        main_assets = data.get("main_assets")  # Get main_assets from request

        if account_name and initial_capital is not None and main_assets:
            account = Account.objects.create(account_name=account_name, initial_capital=initial_capital, main_assets=main_assets)
            return JsonResponse({"message": "Account created successfully!", "id": account.id}, status=201)
        
        return JsonResponse({"error": "Invalid data"}, status=400)


# Delete an account
@csrf_exempt
def delete_account(request, account_id):
    if request.method == "DELETE":
        account = get_object_or_404(Account, id=account_id)
        account.delete()
        return JsonResponse({"message": "Account deleted successfully!"}, status=200)
    return JsonResponse({"error": "Invalid request method"}, status=405)


@csrf_exempt  # You may want to add CSRF handling or use Django's built-in token authentication
def update_account(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            account_id = data.get('id')
            name = data.get('name')
            initial_capital = data.get('initial_capital')
            main_assets = data.get('main_assets')  # Get the updated main_assets

            # Find the account by ID
            account = Account.objects.get(id=account_id)
            account.account_name = name  # Update account name
            account.initial_capital = initial_capital  # Update initial capital

            # Update main_assets if it exists
            if main_assets:
                account.main_assets = main_assets  # Save the updated main assets

            account.save()

            return JsonResponse({
                'id': account.id,
                'name': account.account_name,
                'initial_capital': account.initial_capital,
                'main_assets': account.main_assets,  # Return updated main assets
            })

        except Account.DoesNotExist:
            return JsonResponse({'error': 'Account not found'}, status=404)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)


@csrf_exempt
def get_trading_analytics(request):
    account_name = request.GET.get('account_name')  # Get account_name from the query params
    print(f'Account name is: {account_name}')
    error_count = 0
    
    try:
        # Fetch the account data
        error_count += 1
        account = Account.objects.get(account_name=account_name)
        error_count += 1
        # Fetch related trades for this account
        trades = AccountTrades.objects.filter(account=account)
        error_count += 1

        
        # Prepare the data for response
        analytics_data = {
            'account_name': account.account_name,
            'main_assets': account.main_assets,
            'initial_capital': account.initial_capital,
            'trades': [{
                'asset': trade.asset,
                'order_type': trade.order_type,
                'amount': trade.amount,
                'outcome': trade.outcome,
                'strategy': trade.strategy,
                'day_of_week_entered': trade.day_of_week_entered,
                'day_of_week_closed': trade.day_of_week_closed,
                'trading_session_entered': trade.trading_session_entered,
                'trading_session_closed': trade.trading_session_closed,
            } for trade in trades]
        }
        print('Test 4')
        error_count += 1

        
        return JsonResponse(analytics_data, safe=False)
    
    except Account.DoesNotExist:
        return JsonResponse({'error': f'Account not found with account name: {account_name} with error count: {error_count}'}, status=404)


@csrf_exempt
def create_new_trade_data(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            account_name = data.get('account_name')
            asset = data.get('asset')
            order_type = data.get('order_type')
            strategy = data.get('strategy')
            day_of_week_entered = data.get('day_of_week_entered')
            trading_session_entered = data.get('trading_session_entered')
            outcome = data.get('outcome')
            amount = data.get('amount')
            emotional_bias = data.get('emotional_bias', '')
            reflection = data.get('reflection', '')
            
            # Retrieve the account based on the account_name
            account = Account.objects.get(account_name=account_name)

            # Create the trade entry with the current timestamp
            trade = AccountTrades.objects.create(
                account=account,
                asset=asset,
                order_type=order_type,
                strategy=strategy,
                day_of_week_entered=day_of_week_entered,
                trading_session_entered=trading_session_entered,
                amount=amount,
                emotional_bias=emotional_bias,
                reflection=reflection,
                outcome=outcome,
                date_entered=now(),  # Save the current timestamp
            )

            return JsonResponse({'message': 'Trade recorded successfully!'}, status=200)

        except Account.DoesNotExist:
            return JsonResponse({'error': 'Account not found.'}, status=404)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

    return JsonResponse({'error': 'Invalid request method.'}, status=405)


@csrf_exempt
def fetch_trading_data(request):
    if request.method == 'GET':
        try:
            # Fetching all trade data (you can filter based on the request if needed)
            trades = AccountTrades.objects.all().values('account__account_name', 'asset', 'order_type', 'strategy', 'day_of_week_entered', 'day_of_week_closed', 'trading_session_entered', 'trading_session_closed', 'outcome', 'amount', 'emotional_bias', 'reflection')
            # Return the data as JSON
            return JsonResponse(list(trades), safe=False)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)


@csrf_exempt
def fetch_account_data(request):
    if request.method == 'GET':
        try:
            # Get the account_name from the request
            account_name = request.GET.get('account_name')
            if account_name:
                # Fetch the account data based on account_name
                account = Account.objects.filter(account_name=account_name).first()
                if account:
                    # Serialize the account data
                    account_data = {
                        'account_name': account.account_name,
                        'main_assets': account.main_assets,
                        'initial_capital': account.initial_capital,
                        'trades': list(
                            account.trades.values(
                                'asset',
                                'order_type',
                                'strategy',
                                'day_of_week_entered',
                                'day_of_week_closed',
                                'trading_session_entered',
                                'trading_session_closed',
                                'outcome',
                                'amount',
                                'emotional_bias',
                                'reflection',
                                'date_entered',  # Add this field
                            )
                        )
                    }
                    return JsonResponse(account_data, safe=False)
                else:
                    return JsonResponse({'error': 'Account not found'}, status=404)
            else:
                return JsonResponse({'error': 'Account name is required'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def time_trading_analytics(request):
    if request.method == 'GET':
        account_name = request.GET.get('account_name')
        time_frame = request.GET.get('time_frame', 'month')  # month, week, day
        start_date = request.GET.get('start_date')
        
        # Get base queryset
        trades = AccountTrades.objects.filter(
            account__account_name=account_name,
            date_entered__gte=start_date
        )
        
        # Calculate basic metrics
        total_trades = trades.count()
        winning_trades = trades.filter(outcome='Profit').count()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_profit = trades.filter(outcome='Profit').aggregate(Sum('amount'))['amount__sum'] or 0
        total_loss = abs(trades.filter(outcome='Loss').aggregate(Sum('amount'))['amount__sum'] or 0)
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Get performance by different dimensions
        performance_by_day = list(trades.values('day_of_week_entered').annotate(
            total=Sum('amount'),
            count=Count('id'),
            win_rate=Count(Case(When(outcome='Profit', then=1))) * 100.0 / Count('id')
        ))
        
        performance_by_session = list(trades.values('trading_session_entered').annotate(
            total=Sum('amount'),
            count=Count('id'),
            win_rate=Count(Case(When(outcome='Profit', then=1))) * 100.0 / Count('id')
        ))
        
        performance_by_asset = list(trades.values('asset').annotate(
            total=Sum('amount'),
            count=Count('id'),
            win_rate=Count(Case(When(outcome='Profit', then=1))) * 100.0 / Count('id')
        ))
        
        performance_by_strategy = list(trades.values('strategy').annotate(
            total=Sum('amount'),
            count=Count('id'),
            win_rate=Count(Case(When(outcome='Profit', then=1))) * 100.0 / Count('id')
        ))
        
        # Time series data based on timeframe
        if time_frame == 'month':
            time_series = trades.annotate(
                period=ExtractMonth('date_entered')
            )
        elif time_frame == 'week':
            time_series = trades.annotate(
                period=ExtractWeek('date_entered')
            )
        else:  # day
            time_series = trades.annotate(
                period=F('date_entered__date')
            )
        
        time_series = list(time_series.values('period').annotate(
            total=Sum('amount'),
            count=Count('id'),
            win_rate=Count(Case(When(outcome='Profit', then=1))) * 100.0 / Count('id')
        ).order_by('period'))
        
        response_data = {
            'summary': {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_profit': total_profit,
                'average_win': trades.filter(outcome='Profit').aggregate(Avg('amount'))['amount__avg'],
                'average_loss': trades.filter(outcome='Loss').aggregate(Avg('amount'))['amount__avg'],
            },
            'by_day': performance_by_day,
            'by_session': performance_by_session,
            'by_asset': performance_by_asset,
            'by_strategy': performance_by_strategy,
            'time_series': time_series,
        }
        
        return JsonResponse(response_data)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)


def obtain_dataset(asset, interval, num_days):
    # Calculate the end and start dates
    import datetime
    end_date = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.datetime.now() - datetime.timedelta(days=num_days)).strftime("%Y-%m-%d")

    # Download data using yfinance
    forex_asset = f"{asset}=X"
    data = yf.download(forex_asset, start=start_date, end=end_date, interval=interval)
    return data

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd


def generate_candlestick_chart(data, save_path="candlestick_chart.png"):
    try:
        # Ensure the data has required columns and clean up
        data = data[['Open', 'High', 'Low', 'Close']]

        fig, ax = plt.subplots(figsize=(10, 6))

        for idx, row in enumerate(data.itertuples(index=False)):
            # Access the values by position
            open_price = row[0]
            high_price = row[1]
            low_price = row[2]
            close_price = row[3]

            # Determine the color of the candlestick
            color = 'green' if close_price > open_price else 'red'

            # Draw the candlestick body (rectangle)
            body = Rectangle(
                (idx - 0.4, min(open_price, close_price)),  # Bottom-left corner
                0.8,  # Width
                abs(close_price - open_price),  # Height
                color=color
            )
            ax.add_patch(body)

            # Draw the wick (high-low line)
            ax.plot(
                [idx, idx],  # X-coordinates
                [low_price, high_price],  # Y-coordinates
                color=color
            )

        # Set labels and title
        ax.set_title("Candlestick Chart", fontsize=16)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Price", fontsize=12)

        # Remove x-axis labels
        ax.set_xticks([])

        plt.tight_layout()

        # Save the chart
        plt.savefig(save_path)
        plt.close(fig)  # Close the figure to free up memory

        print(f"Chart saved at: {save_path}")
        return save_path

    except Exception as e:
        print(f"Error generating candlestick chart: {e}")
        return None



def analyse_image_from_file(image_path, news_data):
    try:
        # Read the image and encode in base64
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        return analyse_image(image_data, news_data)
    except Exception as e:
        print(f"Error in image analysis from file: {e}")





@dataclass
class TraderMessage:
    trader_id: str
    content: str
    message_type: str = "discussion"  # Can be "discussion" or "consensus"
    responding_to: Optional[str] = None

class ChartAnnotator:
    def __init__(self, data: pd.DataFrame, fig_size: Tuple[int, int] = (12, 8)):
        self.data = data
        self.fig_size = fig_size

    def extract_price_levels(self, consensus_text: str) -> Dict[str, float]:
        """Extract price levels from consensus text using regex."""
        price_patterns = {
            'support': r'support.*?(\d+\.?\d*)',
            'resistance': r'resistance.*?(\d+\.?\d*)',
            'entry': r'entry.*?(\d+\.?\d*)',
            'stop_loss': r'stop[- ]?loss.*?(\d+\.?\d*)',
            'target': r'target.*?(\d+\.?\d*)'
        }

        levels = {}
        for level_type, pattern in price_patterns.items():
            matches = re.findall(pattern, consensus_text.lower())
            if matches:
                levels[level_type] = float(matches[0])

        return levels

    def draw_annotated_chart(self, consensus_text: str, save_path: str = "annotated_chart.png"):
        """Create an annotated candlestick chart based on consensus analysis."""
        # Create figure and axis
        fig, ax = plt.subplots(figsize=self.fig_size)

        # Extract price levels from consensus
        levels = self.extract_price_levels(consensus_text)

        # Plot candlesticks
        for idx, row in enumerate(self.data.itertuples(index=False)):
            open_price, high_price, low_price, close_price = row[0], row[1], row[2], row[3]

            # Candlestick body
            color = 'green' if close_price > open_price else 'red'
            body = Rectangle(
                (idx - 0.4, min(open_price, close_price)),
                0.8,
                abs(close_price - open_price),
                color=color
            )
            ax.add_patch(body)

            # Wick
            ax.plot([idx, idx], [low_price, high_price], color=color)

        # Add horizontal lines for support and resistance
        max_price = self.data['High'].max()
        min_price = self.data['Low'].min()
        price_range = max_price - min_price
        x_range = len(self.data)

        # Plot levels with different styles
        level_styles = {
            'support': {'color': 'green', 'linestyle': '--', 'alpha': 0.6, 'label': 'Support'},
            'resistance': {'color': 'red', 'linestyle': '--', 'alpha': 0.6, 'label': 'Resistance'},
            'entry': {'color': 'blue', 'linestyle': '-', 'alpha': 0.8, 'label': 'Entry'},
            'stop_loss': {'color': 'red', 'linestyle': ':', 'alpha': 0.8, 'label': 'Stop Loss'},
            'target': {'color': 'green', 'linestyle': ':', 'alpha': 0.8, 'label': 'Target'}
        }

        for level_type, price in levels.items():
            if level_type in level_styles:
                style = level_styles[level_type]
                ax.axhline(y=price, **style)

                # Add price label
                ax.text(x_range + 0.5, price, f'{level_type.replace("_", " ").title()}: {price:.4f}',
                       verticalalignment='center')

        # Add trend arrows if mentioned in consensus
        if 'uptrend' in consensus_text.lower():
            self._add_trend_arrow(ax, 'up', x_range)
        elif 'downtrend' in consensus_text.lower():
            self._add_trend_arrow(ax, 'down', x_range)

        # Formatting
        ax.set_title("Consensus Trading Analysis", fontsize=16)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Price", fontsize=12)
        ax.set_xticks(range(len(self.data.index)))
        ax.set_xticklabels(self.data.index.strftime('%Y-%m-%d'), rotation=45, ha='right')

        # Add legend
        ax.legend()

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

        return save_path

    def _add_trend_arrow(self, ax, direction: str, x_range: int):
        """Add trend arrow to the chart."""
        y_range = ax.get_ylim()
        y_mid = (y_range[0] + y_range[1]) / 2
        arrow_length = x_range * 0.2

        if direction == 'up':
            dx = arrow_length
            dy = (y_range[1] - y_range[0]) * 0.1
            color = 'green'
        else:
            dx = arrow_length
            dy = -(y_range[1] - y_range[0]) * 0.1
            color = 'red'

        arrow = Arrow(x_range * 0.1, y_mid, dx, dy,
                     width=arrow_length * 0.1,
                     color=color, alpha=0.5)
        ax.add_patch(arrow)

def get_economic_events_for_currency(currency_code):
    """Get recent economic events for a specified currency."""
    try:
        # Get events from the last 30 days
        thirty_days_ago = timezone.now() - timezone.timedelta(days=90)
        events = EconomicEvent.objects.filter(
            currency=currency_code,
            date_time__gte=thirty_days_ago
        ).order_by('-date_time')
        
        if not events:
            return "No recent economic events found for this currency."
        
        # Format the events data
        events_text = f"Recent Economic Events for {currency_code}:\n\n"
        for event in events:
            impact_symbol = "🔴" if event.impact == "high" else "🟠" if event.impact == "medium" else "🟢"
            events_text += f"Date: {event.date_time.strftime('%Y-%m-%d %H:%M')}\n"
            events_text += f"Event: {event.event_name} {impact_symbol}\n"
            events_text += f"Actual: {event.actual or 'N/A'}\n"
            events_text += f"Forecast: {event.forecast or 'N/A'}\n"
            events_text += f"Previous: {event.previous or 'N/A'}\n\n"
        
        return events_text
    
    except Exception as e:
        return f"Error retrieving economic events for {currency_code}: {str(e)}"


def extract_currencies_from_pair(forex_pair):
    """Extract base and quote currencies from a forex pair."""
    # Remove any '=X' suffix that might be added for yfinance
    clean_pair = forex_pair.replace('=X', '')
    
    # Dictionary mapping common forex pairs to their base and quote currencies
    forex_pairs_map = {
        'EURUSD': ('EUR', 'USD'),
        'GBPUSD': ('GBP', 'USD'),
        'USDJPY': ('USD', 'JPY'),
        'AUDUSD': ('AUD', 'USD'),
        'USDCHF': ('USD', 'CHF'),
        'NZDUSD': ('NZD', 'USD'),
        'USDCAD': ('USD', 'CAD'),
        'EURJPY': ('EUR', 'JPY'),
        'GBPJPY': ('GBP', 'JPY'),
        'EURGBP': ('EUR', 'GBP'),
        'EURAUD': ('EUR', 'AUD'),
        'EURCAD': ('EUR', 'CAD'),
        'EURCHF': ('EUR', 'CHF'),
        'GBPAUD': ('GBP', 'AUD'),
        'GBPCAD': ('GBP', 'CAD'),
        'GBPCHF': ('GBP', 'CHF'),
        'AUDCAD': ('AUD', 'CAD'),
        'AUDCHF': ('AUD', 'CHF'),
        'AUDJPY': ('AUD', 'JPY'),
        'CADCHF': ('CAD', 'CHF'),
        'CADJPY': ('CAD', 'JPY'),
        'CHFJPY': ('CHF', 'JPY'),
        'NZDCAD': ('NZD', 'CAD'),
        'NZDCHF': ('NZD', 'CHF'),
        'NZDJPY': ('NZD', 'JPY')
    }
    
    if clean_pair in forex_pairs_map:
        return forex_pairs_map[clean_pair]
    else:
        # Fallback: assume first 3 chars are base, last 3 are quote
        if len(clean_pair) >= 6:
            return (clean_pair[:3], clean_pair[3:6])
        else:
            return ('USD', 'USD')  # Default fallback


def get_economic_events_for_pair(forex_pair):
    """Get economic events for both currencies in a forex pair."""
    try:
        base_currency, quote_currency = extract_currencies_from_pair(forex_pair)
        
        # Get events for both currencies
        base_events = get_economic_events_for_currency(base_currency)
        quote_events = get_economic_events_for_currency(quote_currency)
        
        # Combine the events
        combined_events = f"=== ECONOMIC EVENTS FOR {forex_pair} ===\n\n"
        combined_events += f"BASE CURRENCY ({base_currency}):\n"
        combined_events += base_events + "\n"
        combined_events += f"QUOTE CURRENCY ({quote_currency}):\n"
        combined_events += quote_events + "\n"
        
        return combined_events
    
    except Exception as e:
        return f"Error retrieving economic events for pair {forex_pair}: {str(e)}"



@csrf_exempt
def fetch_news_data_api(request):
    """Enhanced news data function that includes economic events."""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests allowed'}, status=405)

    try:
        body = json.loads(request.body)
        assets_to_fetch = body.get('assets', [])
        user_email = body.get('user_email', None)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    if not assets_to_fetch or not user_email:
        return JsonResponse({'error': 'Missing assets or user_email'}, status=400)

    all_news_data = []

    conn = http.client.HTTPSConnection('api.marketaux.com')
    params_template = {
        'api_token': 'xH2KZ1sYqHmNRpfBVfb9C1BbItHMtlRIdZQoRlYw',
        'langauge': 'en',
        'limit': 3,
    }

    for asset in assets_to_fetch:
        params = params_template.copy()
        params['symbols'] = asset
        conn.request('GET', '/v1/news/all?{}'.format(urllib.parse.urlencode(params)))
        res = conn.getresponse()
        data = res.read().decode('utf-8')
        news_data = json.loads(data)

        for article in news_data.get('data', []):
            # Extract highlights properly
            highlights = ''
            if article.get('entities'):
                entity_highlights = article.get('entities')[0].get('highlights', '')
                # Handle different highlight formats
                if isinstance(entity_highlights, str):
                    highlights = entity_highlights
                elif isinstance(entity_highlights, dict):
                    # Extract the actual highlight text
                    highlights = entity_highlights.get('highlight', '') or entity_highlights.get('text', '')
                elif isinstance(entity_highlights, list) and len(entity_highlights) > 0:
                    highlights = entity_highlights[0] if isinstance(entity_highlights[0], str) else str(entity_highlights[0])
            
            news_entry_data = {
                'asset': asset,
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'source': article.get('source', ''),
                'url': article.get('url', ''),
                'highlights': highlights,  # Now always a string
            }
            all_news_data.append(news_entry_data)

    economic_events_data = []
    for asset in assets_to_fetch:
        economic_events = get_economic_events_for_pair(asset)
        economic_events_data.append({
            'asset': asset,
            'economic_events': economic_events
        })

    return JsonResponse({
        'message': all_news_data,
        'economic_events': economic_events_data
    }, safe=False)



def fetch_news_data(assets, user_email):
    """Enhanced news data function that includes economic events."""
    all_news_data = []

    # List of assets to fetch news data for
    assets_to_fetch = assets

    # Establish a connection to the API
    conn = http.client.HTTPSConnection('api.marketaux.com')

    # Define query parameters
    params_template = {
        'api_token': 'xH2KZ1sYqHmNRpfBVfb9C1BbItHMtlRIdZQoRlYw',
        'langauge': 'en',
        'limit': 3,
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
                'asset': asset,
                'title': title,
                'description': description,
                'source': source,
                'url': url,
                'highlights': highlights,
            }
            all_news_data.append(news_entry_data)

    # Add economic events data for each asset
    economic_events_data = []
    for asset in assets_to_fetch:
        economic_events = get_economic_events_for_pair(asset)
        economic_events_data.append({
            'asset': asset,
            'economic_events': economic_events
        })

    return {
        'message': all_news_data,
        'economic_events': economic_events_data
    }


def analyse_image(image_data, news_data):
    """Enhanced image analysis function that includes economic events."""
    try:
        # Getting the base64 string
        base64_image = base64.b64encode(image_data).decode('utf-8')

        api_key = os.environ['OPENAI_API_KEY']

        # Extract discussion prompt if it exists
        discussion_prompt = news_data.get('discussion_prompt', '')

        # Extract economic events data
        economic_events_text = ""
        if 'economic_events' in news_data:
            for event_data in news_data['economic_events']:
                economic_events_text += event_data['economic_events'] + "\n"

        # Extract regular news data
        regular_news_text = ""
        if 'message' in news_data:
            for news_item in news_data['message']:
                regular_news_text += f"Title: {news_item['title']}\n"
                regular_news_text += f"Description: {news_item['description']}\n"
                regular_news_text += f"Source: {news_item['source']}\n"
                if news_item['highlights']:
                    regular_news_text += f"Highlights: {news_item['highlights']}\n"
                regular_news_text += "\n"

        # Construct a more interactive prompt
        prompt = f"""
        {discussion_prompt}

        Based on the trading chart image and the following fundamental data, provide an analysis that addresses the above context.

        ECONOMIC EVENTS DATA:
        {economic_events_text}

        RECENT NEWS DATA:
        {regular_news_text}

        Your response should be in JSON format with two keys:
        1. 'analysis': A detailed analysis that:
           - Incorporates both technical chart analysis and fundamental economic events
           - Directly responds to any previous trader's points if they exist
           - Explains how recent economic events might impact price action
           - Considers the timing and impact level of economic releases
           - Points out any correlation between economic events and chart patterns
           - Explains your reasoning for agreeing or disagreeing with previous analyses
        2. 'recommendation': Either 'buy', 'sell', or 'neutral'

        Make sure to format as valid JSON and avoid line breaks in the text.
        Pay special attention to high-impact economic events (🔴) as they can significantly move the market.
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
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
            "max_tokens": 1500  # Increased to accommodate economic analysis
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        json_data = response.json()
        final_response = json_data['choices'][0]['message']['content']
        return final_response

    except Exception as e:
        print(f"Error occurred in analyse image function: {e}")
        return json.dumps({
            "analysis": f"Error occurred in analysis: {str(e)}",
            "recommendation": "neutral"
        })


def tradergpt(asset, interval, num_days, user_email):
    """Enhanced tradergpt function with economic events integration."""
    try:
        # Step 1: Fetch dataset and generate chart
        data = obtain_dataset(asset, interval, num_days)
        chart_path = generate_candlestick_chart(data)

        # Step 2: Fetch enhanced news data (including economic events)
        news_data = fetch_news_data([asset], user_email)

        # Step 3: Analyse chart with economic events
        chart_analysis = analyse_image_from_file(chart_path, news_data)

        # Combine analysis
        return {
            "chart_analysis": chart_analysis,
            "economic_events": news_data.get('economic_events', [])
        }
    except Exception as e:
        print(f"Error in combined analysis: {e}")


class MultiTraderDialogue:
    def __init__(self, trader1_settings: dict, trader2_settings: dict, max_messages: int = 6):
        self.trader1_settings = trader1_settings
        self.trader2_settings = trader2_settings
        self.max_messages = max_messages
        self.messages = []

        # Initialize market data for both traders
        self.trader1_data = obtain_dataset(
            trader1_settings['asset'],
            trader1_settings['interval'],
            trader1_settings['numDays']
        )
        self.trader2_data = obtain_dataset(
            trader2_settings['asset'],
            trader2_settings['interval'],
            trader2_settings['numDays']
        )

        # Generate charts for both traders
        self.trader1_chart = generate_candlestick_chart(self.trader1_data)
        self.trader2_chart = generate_candlestick_chart(self.trader2_data)

        # Initialize enhanced news data for both assets (including economic events)
        assets = list(set([trader1_settings['asset'], trader2_settings['asset']]))
        self.news_data = fetch_news_data(assets, user_email=None)

        # Initialize chart annotators
        self.trader1_annotator = ChartAnnotator(self.trader1_data)
        self.trader2_annotator = ChartAnnotator(self.trader2_data)

        # Define trader personalities based on settings
        self.trader_personalities = {
            "Trader1": {
                "style": trader1_settings.get('style', 'Conservative'),
                "focus": trader1_settings.get('focus', 'long-term trends and fundamental analysis'),
                "risk_tolerance": trader1_settings.get('risk_tolerance', 'low'),
                "settings": trader1_settings,
                "data": self.trader1_data,
                "chart": self.trader1_chart
            },
            "Trader2": {
                "style": trader2_settings.get('style', 'Aggressive'),
                "focus": trader2_settings.get('focus', 'short-term momentum and technical patterns'),
                "risk_tolerance": trader2_settings.get('risk_tolerance', 'high'),
                "settings": trader2_settings,
                "data": self.trader2_data,
                "chart": self.trader2_chart
            }
        }

    def _create_discussion_prompt(self, trader_id: str, previous_message: Optional[TraderMessage] = None) -> str:
        personality = self.trader_personalities[trader_id]
        settings = personality['settings']
        data = personality['data']

        # Get economic events for this trader's asset
        economic_events_for_asset = ""
        for event_data in self.news_data.get('economic_events', []):
            if event_data['asset'] == settings['asset']:
                economic_events_for_asset = event_data['economic_events']
                break

        base_prompt = f"""
        As a {personality['style']} trader with {personality['risk_tolerance']} risk tolerance,
        analyzing {settings['asset']} on the {settings['interval']} timeframe 
        with a {settings['numDays']}-day lookback period, focusing on {personality['focus']},
        and considering both technical analysis and fundamental economic events,
        """

        if previous_message:
            base_prompt += f"""
            analyze this chart and respond to the following analysis from another trader:

            Previous Analysis: {previous_message.content}

            Consider:
            1. What points do you agree with and why?
            2. What factors might need additional consideration?
            3. How does your timeframe and trading style inform your perspective?
            4. Are there any differences in market behavior between your timeframe and the other trader's timeframe?
            5. How does your risk tolerance affect your view of the suggested trades?
            6. How do the recent economic events support or contradict the previous analysis?
            7. Are there any upcoming economic releases that could impact the trade setup?

            ECONOMIC EVENTS CONTEXT:
            {economic_events_for_asset}

            Aim to find common ground while highlighting important considerations from your perspective.
            Pay special attention to how economic fundamentals align with or contradict technical patterns.
            """
        else:
            base_prompt += f"""
            provide your initial analysis of this chart, incorporating both technical and fundamental analysis.

            ECONOMIC EVENTS CONTEXT:
            {economic_events_for_asset}

            Consider how recent economic events might have influenced the current price action and 
            what they suggest for future price movements.
            """

        # Add market context
        current_close = data['Close'].iloc[-1].item()
        current_open = data['Open'].iloc[-1].item()
        price_change = current_close - current_open
        price_change_pct = (price_change / current_open) * 100

        market_context = f"""
        Current market context for {settings['asset']} ({settings['interval']} timeframe):
        - Price change: {price_change_pct:.2f}%
        - Current price: {current_close:.4f}
        """

        return base_prompt + "\n" + market_context

    def _create_consensus_prompt(self) -> str:
        previous_analyses = "\n".join([
            f"{msg.trader_id} ({self.trader_personalities[msg.trader_id]['settings']['interval']} timeframe, {self.trader_personalities[msg.trader_id]['style']} style): {msg.content}"
            for msg in self.messages if msg.trader_id in ['Trader1', 'Trader2']
        ])

        # Combine all economic events data
        all_economic_events = ""
        for event_data in self.news_data.get('economic_events', []):
            all_economic_events += f"\n{event_data['asset']}:\n{event_data['economic_events']}\n"

        return f"""
        Review the following discussion about market analysis from different timeframes and trading styles:

        {previous_analyses}

        COMBINED ECONOMIC EVENTS DATA:
        {all_economic_events}

        As a group of traders analyzing multiple timeframes and incorporating fundamental analysis, 
        we need to reach a final consensus that considers both technical and fundamental factors.

        Please provide specific levels in your analysis:
        1. Key support and resistance levels from both timeframes
        2. Suggested entry price considering both analyses and economic events
        3. Stop-loss level that respects both timeframes and fundamental risks
        4. Target price based on both technical and fundamental perspectives
        5. Overall trend direction (uptrend/downtrend) on each timeframe
        6. Economic event risks and opportunities

        Also include:
        1. Points of agreement between different timeframe analyses
        2. How different risk tolerances and trading styles are balanced
        3. How technical analysis aligns with or contradicts fundamental economic data
        4. Final recommendation that considers both timeframes and economic events
        5. Risk management suggestions that account for both technical and fundamental risks
        6. Timing considerations based on upcoming economic releases

        Format your response as JSON with 'analysis' and 'recommendation' keys.
        Include numerical price levels in your analysis for chart annotation.
        Give special weight to high-impact economic events (🔴) in your analysis.
        """

    def _generate_response(self, trader_id: str, previous_message: Optional[TraderMessage] = None,
                         message_type: str = "discussion") -> str:
        if message_type == "consensus":
            prompt = self._create_consensus_prompt()
            chart_path = self.trader1_chart
        else:
            prompt = self._create_discussion_prompt(trader_id, previous_message)
            chart_path = self.trader_personalities[trader_id]['chart']

        modified_news = self.news_data.copy()
        modified_news['discussion_prompt'] = prompt

        return analyse_image_from_file(chart_path, modified_news)

    def conduct_dialogue(self) -> Tuple[List[TraderMessage], str]:
        # First message from Trader1
        initial_message = TraderMessage(
            trader_id="Trader1",
            content=self._generate_response("Trader1"),
            message_type="discussion"
        )
        self.messages.append(initial_message)

        # Continue discussion
        current_msg_count = 1
        discussion_messages = self.max_messages - 1

        while current_msg_count < discussion_messages:
            current_trader = "Trader2" if current_msg_count % 2 == 1 else "Trader1"
            previous_message = self.messages[-1]

            response = TraderMessage(
                trader_id=current_trader,
                content=self._generate_response(current_trader, previous_message),
                message_type="discussion",
                responding_to=previous_message.trader_id
            )
            self.messages.append(response)
            current_msg_count += 1

        # Final consensus
        consensus = TraderMessage(
            trader_id="Consensus",
            content=self._generate_response(
                trader_id="Consensus",
                message_type="consensus"
            ),
            message_type="consensus"
        )
        self.messages.append(consensus)

        # Create annotated chart based on consensus
        annotated_chart_path = self.trader1_annotator.draw_annotated_chart(
            consensus.content,
            save_path=f"annotated_multi_timeframe_chart.png"
        )

        return self.messages, annotated_chart_path


@csrf_exempt
def get_trader_analysis(request):
    """Enhanced trader analysis endpoint with economic events integration."""
    try:
        if request.method == 'POST':
            data = json.loads(request.body)
            traders_settings = data.get('traders', {})
            
            # Validate trader settings
            if 'trader1' not in traders_settings or 'trader2' not in traders_settings:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Settings for both traders are required.',
                    'type': 'ValidationError'
                }, status=400)
            
            # Initialize multi-trader dialogue with enhanced economic events
            dialogue = MultiTraderDialogue(
                trader1_settings=traders_settings['trader1'],
                trader2_settings=traders_settings['trader2']
            )
            
            # Run the dialogue
            conversation, chart_path = dialogue.conduct_dialogue()
            
            # Convert the conversation to a serializable format
            conversation_data = []
            for msg in conversation:
                if isinstance(msg.content, str):
                    content = msg.content.replace('```json\n', '').replace('\n```', '')
                    try:
                        parsed_content = json.loads(content)
                        if 'analysis' in parsed_content:
                            if isinstance(parsed_content['analysis'], str):
                                parsed_content['analysis'] = parsed_content['analysis'][:2000]  # Increased limit for economic analysis
                            else:
                                parsed_content['analysis'] = str(parsed_content['analysis'])[:2000]
                        content = parsed_content
                    except json.JSONDecodeError:
                        content = content[:2000]  # Increased limit for economic analysis
                else:
                    content = msg.content

                conversation_data.append({
                    'trader_id': msg.trader_id,
                    'content': content,
                    'message_type': msg.message_type,
                    'responding_to': msg.responding_to,
                    'settings': {
                        'asset': dialogue.trader_personalities[msg.trader_id]['settings']['asset'] if msg.trader_id in dialogue.trader_personalities else None,
                        'interval': dialogue.trader_personalities[msg.trader_id]['settings']['interval'] if msg.trader_id in dialogue.trader_personalities else None,
                        'numDays': dialogue.trader_personalities[msg.trader_id]['settings']['numDays'] if msg.trader_id in dialogue.trader_personalities else None,
                        'style': dialogue.trader_personalities[msg.trader_id]['style'] if msg.trader_id in dialogue.trader_personalities else None,
                        'focus': dialogue.trader_personalities[msg.trader_id]['focus'] if msg.trader_id in dialogue.trader_personalities else None,
                        'risk_tolerance': dialogue.trader_personalities[msg.trader_id]['risk_tolerance'] if msg.trader_id in dialogue.trader_personalities else None,
                    } if msg.trader_id != 'Consensus' else None
                })

            # Process the chart image
            image = Image.open(chart_path)
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            max_size = (800, 800)
            image.thumbnail(max_size)
            
            compressed_image_io = io.BytesIO()
            image.save(compressed_image_io, format='JPEG', quality=50)
            compressed_image_io.seek(0)
            
            encoded_image = base64.b64encode(compressed_image_io.read()).decode('utf-8')
            
            if os.path.exists(chart_path):
                os.remove(chart_path)
            
            response_data = {
                'status': 'success',
                'conversation': conversation_data,
                'chart_image': encoded_image,
                'analysis_summary': {
                    'trader1': traders_settings['trader1'],
                    'trader2': traders_settings['trader2']
                },
                'economic_events_included': True,  # Flag to indicate economic events are included
                'currencies_analyzed': []  # Will be populated with currency pairs
            }
            
            # Add information about which currencies were analyzed
            for asset in [traders_settings['trader1']['asset'], traders_settings['trader2']['asset']]:
                base_curr, quote_curr = extract_currencies_from_pair(asset)
                response_data['currencies_analyzed'].append({
                    'pair': asset,
                    'base_currency': base_curr,
                    'quote_currency': quote_curr
                })
            
            return JsonResponse(response_data)
        else:
            return JsonResponse({
                'status': 'error',
                'message': 'Invalid request method.',
                'type': 'InvalidRequestMethod'
            }, status=400)
            
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e),
            'type': type(e).__name__
        }, status=500)


# Additional utility function to help with currency mapping updates
def update_currency_mapping():
    """Utility function to add more currency pairs to the mapping if needed."""
    additional_pairs = {
        'GBPNZD': ('GBP', 'NZD'),
        'AUDNZD': ('AUD', 'NZD'),
        'EURCZK': ('EUR', 'CZK'),
        'USDPLN': ('USD', 'PLN'),
        'USDHUF': ('USD', 'HUF'),
        'USDTRY': ('USD', 'TRY'),
        'USDZAR': ('USD', 'ZAR'),
        'USDMXN': ('USD', 'MXN'),
        'USDSEK': ('USD', 'SEK'),
        'USDNOK': ('USD', 'NOK'),
        'USDDKK': ('USD', 'DKK')
    }
    
    return additional_pairs

def bullish_market_sentiment(asset):
    
    news_data = fetch_news_data([asset], None)
    prompt = f'''
    Please give me the sentiment reading for this news data in one word.

    Either Bullish, Bearish or Neutral

    {news_data}

    '''
    sentiment = chat_gpt(prompt)
    if sentiment.lower() == 'bullish':
        return True
    else:
        return False


def bearish_market_sentiment(asset):

    news_data = fetch_news_data([asset], None)
    prompt = f'''
    Please give me the sentiment reading for this news data in one word.

    Either Bullish, Bearish or Neutral

    {news_data}

    '''
    sentiment = chat_gpt(prompt)
    if sentiment.lower() == 'bearish':
        return True
    else:
        return False


@csrf_exempt
def save_backtest_model_data(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            backtest = BacktestModels.objects.create(
                chosen_dataset=data.get('chosen_dataset'),
                generated_code=data.get('generated_code'),
                model_backtested=data.get('model_backtested', False),
                dataset_start=data.get('dataset_start'),
                dataset_end=data.get('dataset_end'),
                initial_capital=data.get('initial_capital')
            )
            return JsonResponse({'message': 'Backtest data saved successfully!', 'id': backtest.id}, status=201)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'error': 'Invalid request method'}, status=405)


@csrf_exempt
def generate_idea(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            # Create a new idea
            idea = IdeaModel.objects.create(
                idea_category=data.get('idea_category'),
                idea_text=data.get('idea_text'),
                idea_tracker=data.get('idea_tracker', 'Pending')
            )
            
            return JsonResponse({
                'success': True,
                'message': 'Idea created successfully',
                'idea': {
                    'id': idea.id,
                    'idea_category': idea.idea_category,
                    'idea_text': idea.idea_text,
                    'idea_tracker': idea.idea_tracker,
                    'created_at': idea.created_at.isoformat()
                }
            })
        except Exception as e:
            return JsonResponse({
                'success': False,
                'message': str(e)
            }, status=400)
    
    return JsonResponse({
        'success': False,
        'message': 'Invalid request method'
    }, status=405)


@csrf_exempt
def fetch_ideas(request):
    if request.method == 'GET':
        try:
            # Get all ideas, ordered by creation date (newest first)
            ideas = IdeaModel.objects.all().order_by('-created_at')
            
            # Serialize the ideas
            ideas_list = [{
                'id': idea.id,
                'idea_category': idea.idea_category,
                'idea_text': idea.idea_text,
                'idea_tracker': idea.idea_tracker,
                'created_at': idea.created_at.isoformat()
            } for idea in ideas]
            
            return JsonResponse(ideas_list, safe=False)
        except Exception as e:
            return JsonResponse({
                'success': False,
                'message': str(e)
            }, status=500)
    
    return JsonResponse({
        'success': False,
        'message': 'Invalid request method'
    }, status=405)


@csrf_exempt
def delete_idea(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            idea_id = data.get('idea_id')
            
            # Find and delete the idea
            idea = IdeaModel.objects.get(id=idea_id)
            idea.delete()
            
            return JsonResponse({'status': 'success', 'message': 'Idea deleted successfully'})
        except IdeaModel.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Idea not found'}, status=404)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    
    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)


@csrf_exempt
def update_idea_tracker(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            idea_id = data.get('idea_id')
            new_tracker = data.get('idea_tracker')
            
            # Find and update the idea
            idea = IdeaModel.objects.get(id=idea_id)
            idea.idea_tracker = new_tracker
            idea.save()
            
            return JsonResponse({'status': 'success', 'message': 'Idea tracker updated successfully'})
        except IdeaModel.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Idea not found'}, status=404)
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    
    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)


@csrf_exempt
def update_idea(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            idea_id = data.get('idea_id')
            
            # Find the idea by ID
            idea = IdeaModel.objects.get(id=idea_id)
            
            # Update the idea fields
            idea.idea_category = data.get('idea_category', idea.idea_category)
            idea.idea_text = data.get('idea_text', idea.idea_text)
            idea.idea_tracker = data.get('idea_tracker', idea.idea_tracker)
            
            # Save the updated idea
            idea.save()
            
            return JsonResponse({
                'status': 'success', 
                'message': 'Idea updated successfully',
                'idea': {
                    'id': idea.id,
                    'idea_category': idea.idea_category,
                    'idea_text': idea.idea_text,
                    'idea_tracker': idea.idea_tracker,
                    'created_at': idea.created_at.isoformat()
                }
            })
        except IdeaModel.DoesNotExist:
            return JsonResponse({
                'status': 'error', 
                'message': 'Idea not found'
            }, status=404)
        except Exception as e:
            return JsonResponse({
                'status': 'error', 
                'message': str(e)
            }, status=500)
    
    return JsonResponse({
        'status': 'error', 
        'message': 'Method not allowed'
    }, status=405)


@csrf_exempt
def get_ai_account_summary(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            account_name = data.get('account_name')
            metrics = data.get('metrics', {})
            trades = data.get('trades', [])
            
            # Create a prompt for the AI with professional guidance
            prompt = f"""
            Prepare a comprehensive trading performance analysis for account '{account_name}' with the following requirements:
            - Maintain a professional and analytical tone
            - Use clear, concise language
            - Provide objective insights and data-driven recommendations
            
            Performance Metrics Overview:
            - Win Rate: {metrics.get('winRate', 0)}%
            - Average Win: ${metrics.get('averageWin', 0)}
            - Average Loss: ${metrics.get('averageLoss', 0)}
            - Profit Factor: {metrics.get('profitFactor', 0)}
            - Number of Wins: {metrics.get('numberOfWins', 0)}
            - Number of Losses: {metrics.get('numberOfLosses', 0)}

            Trading Data: \n\n{trades}\n\n
            
            Analysis Guidelines:
            1. Deliver a precise performance assessment
            2. Identify key performance trends and patterns
            3. Develop specific, actionable improvement strategies
            4. Maintain a concise format (under 250 words)
            5. Use professional terminology
            6. Highlight both strengths and areas for potential improvement
            7. AVOID USING ASTERISKS, HASHTAGS, OR MARKDOWN FORMATTING

            
            Presentation Instructions:
            - AVOID USING ASTERISKS, HASHTAGS, OR MARKDOWN FORMATTING
            - Use subtle emojis sparingly for visual emphasis 📊
            - Ensure a structured, professional presentation

            AVOID USING ASTERISKS, HASHTAGS, OR MARKDOWN FORMATTING

            """
            
            # Get AI summary
            summary = chat_gpt(prompt)
            
            return JsonResponse({'summary': summary})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=400)


@csrf_exempt
def save_quiz(request):
    if request.method == 'POST':
        try:
            # Parse the incoming JSON data
            data = json.loads(request.body)
            
            # Extract quiz details
            quiz_name = data.get('quiz_name', 'Unnamed Quiz')
            total_questions = data.get('total_questions', 0)
            correct_answers = data.get('correct_answers', 0)
            questions = data.get('questions', [])

            # Create SavedQuiz instance
            saved_quiz = SavedQuiz.objects.create(
                quiz_name=quiz_name,
                total_questions=total_questions,
                correct_answers=correct_answers
            )

            # Create SavedQuizQuestion instances
            for question_data in questions:
                SavedQuizQuestion.objects.create(
                    saved_quiz=saved_quiz,
                    question=question_data.get('question', ''),
                    selected_answer=question_data.get('selectedAnswer', ''),
                    correct_answer=question_data.get('correctAnswer', ''),
                    is_correct=question_data.get('isCorrect', False)
                )

            return JsonResponse({
                'status': 'success', 
                'message': 'Quiz saved successfully',
                'saved_quiz_id': saved_quiz.id
            }, status=201)

        except json.JSONDecodeError:
            return JsonResponse({
                'status': 'error', 
                'message': 'Invalid JSON data'
            }, status=400)
        except Exception as e:
            return JsonResponse({
                'status': 'error', 
                'message': str(e)
            }, status=400)
    
    return JsonResponse({
        'status': 'error', 
        'message': 'Invalid request method'
    }, status=405)


@csrf_exempt
def fetch_saved_quizzes(request):
    if request.method != 'GET':
        return JsonResponse({
            'status': 'error',
            'message': 'Only GET method is allowed'
        }, status=405)  # Method Not Allowed status code

    try:
        # Fetch all saved quizzes, ordered by most recent first
        saved_quizzes = SavedQuiz.objects.order_by('-created_at')
        
        # Prepare the response data
        quizzes_data = []
        for quiz in saved_quizzes:
            # Get associated questions for each quiz
            questions = quiz.questions.all()
            
            quiz_details = {
                'id': quiz.id,
                'quiz_name': quiz.quiz_name,
                'total_questions': quiz.total_questions,
                'correct_answers': quiz.correct_answers,
                'created_at': quiz.created_at.isoformat(),
                'questions': [
                    {
                        'question': q.question,
                        'selected_answer': q.selected_answer,
                        'correct_answer': q.correct_answer,
                        'is_correct': q.is_correct
                    } for q in questions
                ]
            }
            quizzes_data.append(quiz_details)
        
        return JsonResponse({
            'status': 'success',
            'quizzes': quizzes_data
        }, safe=True)
    
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)


@csrf_exempt
def delete_quiz(request, quiz_id):
    """
    Delete a saved quiz by its ID
    """
    if request.method == 'DELETE':
        try:
            # Get the quiz or return 404 if not found
            quiz = get_object_or_404(SavedQuiz, id=quiz_id)
            
            # Delete the quiz (this will also delete related questions due to CASCADE)
            quiz.delete()
            
            return JsonResponse({
                'status': 'success', 
                'message': 'Quiz deleted successfully'
            }, status=200)
        
        except Exception as e:
            return JsonResponse({
                'status': 'error', 
                'message': str(e)
            }, status=500)
    
    return JsonResponse({
        'status': 'error', 
        'message': 'Invalid request method'
    }, status=405)


import logging

logger = logging.getLogger(__name__)

@csrf_exempt
def delete_music(request):
    MusicModel.objects.all().delete()
    return JsonResponse({'message': 'Songs Deleted!'})


@csrf_exempt
def save_music(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed'}, status=405)
    
    try:
        name = request.POST.get('name')
        file = request.FILES.get('file')
        
        if not name or not file:
            return JsonResponse({'error': 'Name and file are required'}, status=400)
        
        # Read the file content
        file_data = file.read()
        file_name = file.name
        content_type = file.content_type or 'audio/mpeg'
        
        # Check if a song with this name already exists
        existing_song = MusicModel.objects.filter(name=name).first()
        if existing_song:
            # If it exists, update the record
            existing_song.file_data = file_data
            existing_song.file_name = file_name
            existing_song.content_type = content_type
            existing_song.save()
            return JsonResponse({'success': True, 'message': f'Updated song: {name}'})
        
        # Create new song record
        music = MusicModel(
            name=name, 
            file_data=file_data,
            file_name=file_name,
            content_type=content_type
        )
        music.save()
        
        return JsonResponse({'success': True, 'message': f'Saved song: {name}'})
    
    except Exception as e:
        logger.error(f"Error saving music: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

from django.urls import reverse


@csrf_exempt
def fetch_music(request):
    try:
        songs = MusicModel.objects.all()
        songs_data = []
        
        for song in songs:
            # Generate a URL to stream this song
            stream_url = request.build_absolute_uri(
                reverse('stream_music', args=[song.id])
            )
            
            songs_data.append({
                'id': song.id,
                'name': song.name,
                'file': stream_url,
                'file_name': song.file_name,
                'updated_at': song.updated_at.isoformat()
            })
        
        return JsonResponse({'songs': songs_data}, safe=False)
    except Exception as e:
        logger.error(f"Error fetching music: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def stream_music(request, song_id):
    try:
        song = get_object_or_404(MusicModel, id=song_id)
        
        # Create a response with the binary data
        response = HttpResponse(song.file_data, content_type=song.content_type)
        response['Content-Disposition'] = f'inline; filename="{song.file_name}"'
        
        return response
    except Exception as e:
        logger.error(f"Error streaming music: {str(e)}")
        return HttpResponse(status=500)        


@csrf_exempt
def fetch_asset_update(request):
    """Fetch current data for all tracked assets"""
    try:
        assets = AssetsTracker.objects.all()
        asset_data = []

        for asset_obj in assets:
            asset = asset_obj.asset
            
            try:
                # Use your existing obtain_dataset function - more reliable
                data = obtain_dataset(asset, "1d", 5)  # Get daily data for the last 5 days
                
                if not data.empty and len(data) >= 2:
                    # Get the most recent price
                    current_price = float(data['Close'].iloc[-1])
                    
                    # Get the previous candle's close for comparison
                    previous_price = float(data['Close'].iloc[-2])
                    
                    # Calculate the percentage change
                    percent_change = round(((current_price - previous_price) / previous_price) * 100, 2)

                    # Format the timestamp in a readable format
                    last_updated = data.index[-1].strftime("%Y-%m-%d %H:%M")

                    asset_data.append({
                        'id': asset_obj.id,
                        'asset': asset,
                        'current_price': round(current_price, 4),
                        'percent_change': percent_change,
                        'last_updated': last_updated
                    })
                else:
                    # Handle case where we don't have enough data
                    asset_data.append({
                        'id': asset_obj.id,
                        'asset': asset,
                        'current_price': 0,
                        'percent_change': 0,
                        'error': "Insufficient data"
                    })
            except Exception as e:
                print(f"Error getting data for {asset}: {str(e)}")
                # Include the asset in the response anyway with default values
                asset_data.append({
                    'id': asset_obj.id,
                    'asset': asset,
                    'current_price': 0,
                    'percent_change': 0,
                    'error': str(e)
                })

        return JsonResponse(asset_data, safe=False)
    except Exception as e:
        print(f"Error in fetch_asset_update: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def get_tracked_assets(request):
    """Get all tracked assets"""
    try:
        assets = AssetsTracker.objects.all()
        asset_data = []
        
        for asset_obj in assets:
            asset = asset_obj.asset
            
            try:
                # Use obtain_dataset function with daily interval to get day-over-day changes
                data = obtain_dataset(asset, "1d", 5)  # Get daily data for the last 5 days
                
                if not data.empty and len(data) >= 2:
                    # Get the most recent price
                    current_price = float(data['Close'].iloc[-1])
                    
                    # Get yesterday's close for day-over-day comparison
                    yesterday_close = float(data['Close'].iloc[-2])
                    
                    # Calculate the percentage change
                    percent_change = round(((current_price - yesterday_close) / yesterday_close) * 100, 2)
                    
                    asset_data.append({
                        'id': asset_obj.id,
                        'asset': asset,
                        'current_price': round(current_price, 4),
                        'percent_change': percent_change,
                        'last_updated': data.index[-1].strftime("%Y-%m-%d")
                    })
                else:
                    # Handle case where we don't have enough data
                    asset_data.append({
                        'id': asset_obj.id,
                        'asset': asset,
                        'current_price': 0,
                        'percent_change': 0,
                        'error': "Insufficient data"
                    })
            except Exception as e:
                print(f"Error getting data for {asset}: {str(e)}")
                # Include the asset in the response anyway with default values
                asset_data.append({
                    'id': asset_obj.id,
                    'asset': asset,
                    'current_price': 0,
                    'percent_change': 0,
                    'error': str(e)
                })
        
        return JsonResponse(asset_data, safe=False)
    except Exception as e:
        print(f"Error in get_tracked_assets: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)
        
        
@csrf_exempt
@require_http_methods(["POST"])
def add_tracked_asset(request):
    """Add a new asset to track"""
    
    data = json.loads(request.body)
    asset = data.get('asset')
        
    if not asset:
        return JsonResponse({'error': 'Asset is required'}, status=400)
        
    # Check if asset already exists
    if AssetsTracker.objects.filter(asset=asset).exists():
        return JsonResponse({'error': 'Asset already tracked'}, status=400)
        
    # Create new asset tracker
    asset_tracker = AssetsTracker.objects.create(asset=asset)
        
    return JsonResponse({
        'id': asset_tracker.id,
        'asset': asset_tracker.asset
    })
    # except Exception as e:
    #     return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def remove_tracked_asset(request):
    """Remove a tracked asset"""
    
    data = json.loads(request.body)
    asset_id = data.get('id')
        
    if not asset_id:
        return JsonResponse({'error': 'Asset ID is required'}, status=400)
        
    # Delete the asset tracker
    AssetsTracker.objects.filter(id=asset_id).delete()
        
    return JsonResponse({'success': True})
    # except Exception as e:
    #     return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def get_trade_ideas(request):
    trade_ideas = TradeIdea.objects.all().order_by('-date_created')
    data = list(trade_ideas.values())
    return JsonResponse({'trade_ideas': data})

@csrf_exempt
@require_http_methods(["POST"])
def create_trade_idea(request):
    try:
        data = json.loads(request.body)
        trade_idea = TradeIdea.objects.create(
            heading=data.get('heading'),
            asset=data.get('asset'),
            trade_idea=data.get('trade_idea'),
            trade_status=data.get('trade_status', 'pending'),
            target_price=data.get('target_price'),
            stop_loss=data.get('stop_loss'),
            entry_price=data.get('entry_price'),
            outcome=data.get('outcome', 'pending')  # Added outcome field
        )
        return JsonResponse({
            'success': True,
            'trade_idea_id': trade_idea.id
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)

@csrf_exempt
@require_http_methods(["PUT"])
def update_trade_idea(request, id):
    try:
        data = json.loads(request.body)
        trade_idea = TradeIdea.objects.get(id=id)
        
        # Update fields if they exist in request
        if 'heading' in data:
            trade_idea.heading = data['heading']
        if 'asset' in data:
            trade_idea.asset = data['asset']
        if 'trade_idea' in data:
            trade_idea.trade_idea = data['trade_idea']
        if 'trade_status' in data:
            trade_idea.trade_status = data['trade_status']
        if 'target_price' in data:
            trade_idea.target_price = data['target_price']
        if 'stop_loss' in data:
            trade_idea.stop_loss = data['stop_loss']
        if 'entry_price' in data:
            trade_idea.entry_price = data['entry_price']
        if 'outcome' in data:  # Added outcome field handling
            trade_idea.outcome = data['outcome']
            
        trade_idea.save()
        
        return JsonResponse({
            'success': True,
            'trade_idea_id': trade_idea.id
        })
    except TradeIdea.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': 'Trade idea not found'
        }, status=404)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)

@csrf_exempt
@require_http_methods(["DELETE"])
def delete_trade_idea(request, id):
    try:
        trade_idea = TradeIdea.objects.get(id=id)
        trade_idea.delete()
        return JsonResponse({
            'success': True
        })
    except TradeIdea.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': 'Trade idea not found'
        }, status=404)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)



from django.conf import settings
from datetime import datetime, date
from decimal import Decimal
import base64
from django.core.files.base import ContentFile

@csrf_exempt
@require_http_methods(["GET"])
def get_prop_firms(request):
    firms = PropFirm.objects.all()
    data = []
    
    for firm in firms:
        firm_data = {
            'id': firm.id,
            'name': firm.name,
            'website': firm.website,
            'description': firm.description
        }
        try:

            if firm.logo:
                # Read the file content directly from the model
                image_file = firm.logo
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                image_type = firm.logo.name.split('.')[-1].lower()
                firm_data['logo'] = f"data:image/{image_type};base64,{encoded_string}"
            else:
                firm_data['logo'] = None
        except Exception as e:
            print(f'Error occured in get_prop_firms function: {e}')
            firm_data['logo'] = None


            
        data.append(firm_data)
    
    return JsonResponse({'firms': data})


@csrf_exempt
@require_http_methods(["POST"])
def create_prop_firm(request):
    try:
        # Check if the request has files
        if request.FILES and 'logo' in request.FILES:
            # Handle form data with file upload
            name = request.POST.get('name')
            website = request.POST.get('website')
            description = request.POST.get('description')
            
            # Create prop firm
            firm = PropFirm.objects.create(
                name=name,
                website=website,
                description=description
            )
            
            # Handle logo upload
            firm.logo = request.FILES['logo']
            firm.save()
        else:
            # Handle JSON data without file upload
            data = json.loads(request.body)
            name = data.get('name')
            website = data.get('website')
            description = data.get('description')
            
            # Create prop firm
            firm = PropFirm.objects.create(
                name=name,
                website=website,
                description=description
            )
        
        return JsonResponse({
            'success': True,
            'firm': {
                'id': firm.id,
                'name': firm.name,
                'logo': firm.logo.url if firm.logo else None,
                'website': firm.website,
                'description': firm.description
            }
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@csrf_exempt
@require_http_methods(["GET"])
def get_prop_accounts(request):
    accounts = PropFirmAccount.objects.all()
    data = []
    
    for account in accounts:
        account_data = {
            'id': account.id,
            'prop_firm': {
                'id': account.prop_firm.id,
                'name': account.prop_firm.name,
                'logo': account.prop_firm.logo.url if account.prop_firm.logo else None
            },
            'account_name': account.account_name,
            'account_id': account.account_id,
            'account_type': account.account_type,
            'status': account.status,
            'initial_balance': float(account.initial_balance),
            'current_balance': float(account.current_balance),
            'current_equity': float(account.current_equity),
            'daily_loss_limit': float(account.daily_loss_limit) if account.daily_loss_limit else None,
            'max_loss_limit': float(account.max_loss_limit) if account.max_loss_limit else None,
            'profit_target': float(account.profit_target) if account.profit_target else None,
            'start_date': account.start_date.isoformat(),
            'end_date': account.end_date.isoformat() if account.end_date else None,
            'days_remaining': account.days_remaining(),
            'percentage_to_target': account.percentage_to_target(),
        }
        data.append(account_data)
    
    return JsonResponse({'accounts': data})


@csrf_exempt
@require_http_methods(["POST"])
def create_prop_account(request):
    try:
        data = json.loads(request.body)
        
        prop_firm = PropFirm.objects.get(id=data.get('prop_firm_id'))
        
        account = PropFirmAccount.objects.create(
            prop_firm=prop_firm,
            account_name=data.get('account_name'),
            account_id=data.get('account_id'),
            account_type=data.get('account_type'),
            status=data.get('status', 'IN_PROGRESS'),
            initial_balance=Decimal(data.get('initial_balance')),
            current_balance=Decimal(data.get('initial_balance')),  # Start with initial balance
            current_equity=Decimal(data.get('initial_balance')),   # Start with initial balance
            daily_loss_limit=Decimal(data.get('daily_loss_limit')) if data.get('daily_loss_limit') else None,
            max_loss_limit=Decimal(data.get('max_loss_limit')) if data.get('max_loss_limit') else None,
            profit_target=Decimal(data.get('profit_target')) if data.get('profit_target') else None,
            start_date=datetime.strptime(data.get('start_date'), '%Y-%m-%d').date(),
            end_date=datetime.strptime(data.get('end_date'), '%Y-%m-%d').date() if data.get('end_date') else None,
        )
        
        # Update metrics
        update_prop_metrics()
        
        return JsonResponse({
            'success': True,
            'account': {
                'id': account.id,
                'account_name': account.account_name
            }
        })
    except PropFirm.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Prop firm not found'}, status=404)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@csrf_exempt
@require_http_methods(["POST"])
def update_prop_account_balance(request, account_id):
    try:
        data = json.loads(request.body)
        account = PropFirmAccount.objects.get(id=account_id)
        
        account.current_balance = Decimal(data.get('current_balance'))
        account.current_equity = Decimal(data.get('current_equity'))
        account.save()
        
        # Update metrics
        update_prop_metrics()
        
        return JsonResponse({'success': True})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@csrf_exempt
@require_http_methods(["POST"])
def add_prop_trading_day(request, account_id):
    try:
        # Check if the request has form data or JSON
        if request.content_type and 'multipart/form-data' in request.content_type:
            # Handle form data
            account = PropFirmAccount.objects.get(id=account_id)
            
            # Create trading day
            trading_day = TradingDay.objects.create(
                account=account,
                date=datetime.strptime(request.POST.get('date'), '%Y-%m-%d').date(),
                starting_balance=Decimal(request.POST.get('starting_balance')),
                ending_balance=Decimal(request.POST.get('ending_balance')),
                pnl=Decimal(request.POST.get('pnl')),
                session_time_minutes=int(request.POST.get('session_time_minutes', 0)),
                notes=request.POST.get('notes')
            )
            
            # Handle voice memo if included
            if 'voice_memo' in request.FILES:
                trading_day.voice_memo = request.FILES['voice_memo']
                trading_day.save()
        else:
            # Handle JSON data
            data = json.loads(request.body)
            account = PropFirmAccount.objects.get(id=account_id)
            
            # Create trading day
            trading_day = TradingDay.objects.create(
                account=account,
                date=datetime.strptime(data.get('date'), '%Y-%m-%d').date(),
                starting_balance=Decimal(data.get('starting_balance')),
                ending_balance=Decimal(data.get('ending_balance')),
                pnl=Decimal(data.get('pnl')),
                session_time_minutes=int(data.get('session_time_minutes', 0)),
                notes=data.get('notes')
            )
        
        # Update account balance if it's the most recent day
        if trading_day.date == date.today():
            account.current_balance = trading_day.ending_balance
            account.current_equity = trading_day.ending_balance  # Simplified, in real app equity might differ
            account.save()
        
        # Update metrics
        update_prop_metrics()
        
        return JsonResponse({'success': True, 'trading_day_id': trading_day.id})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@csrf_exempt
@require_http_methods(["POST"])
def add_prop_trade(request, account_id):
    try:
        data = json.loads(request.body)
        account = PropFirmAccount.objects.get(id=account_id)
        
        # Get trading day if specified
        trading_day = None
        if data.get('trading_day_id'):
            trading_day = TradingDay.objects.get(id=data.get('trading_day_id'), account=account)
        
        # Calculate PnL if exit price is provided
        pnl = None
        if data.get('exit_price'):
            price_diff = Decimal(data.get('exit_price')) - Decimal(data.get('entry_price'))
            if data.get('trade_type') == 'SELL':
                price_diff = -price_diff
            pnl = price_diff * Decimal(data.get('size'))
        
        # Create trade
        trade = PropTrade.objects.create(
            account=account,
            trading_day=trading_day,
            asset=data.get('asset'),
            trade_type=data.get('trade_type'),
            entry_price=Decimal(data.get('entry_price')),
            exit_price=Decimal(data.get('exit_price')) if data.get('exit_price') else None,
            size=Decimal(data.get('size')),
            entry_time=datetime.strptime(data.get('entry_time'), '%Y-%m-%dT%H:%M:%S'),
            exit_time=datetime.strptime(data.get('exit_time'), '%Y-%m-%dT%H:%M:%S') if data.get('exit_time') else None,
            pnl=pnl,
            strategy=data.get('strategy'),
            notes=data.get('notes')
        )
        
        # Update metrics
        update_prop_metrics()
        
        return JsonResponse({'success': True, 'trade_id': trade.id})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@csrf_exempt
@require_http_methods(["GET"])
def get_prop_account_analytics(request, account_id):
    try:
        account = PropFirmAccount.objects.get(id=account_id)
        
        # Get trading days
        trading_days = TradingDay.objects.filter(account=account).order_by('date')
        trading_days_data = [{
            'date': day.date.isoformat(),
            'pnl': float(day.pnl),
            'balance': float(day.ending_balance)
        } for day in trading_days]
        
        # Get trades
        trades = PropTrade.objects.filter(account=account, exit_time__isnull=False)
        
        # Calculate analytics
        total_trades = trades.count()
        winning_trades = trades.filter(pnl__gt=0).count()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = trades.filter(pnl__gt=0).aggregate(avg=models.Avg('pnl'))['avg'] or 0
        avg_loss = trades.filter(pnl__lt=0).aggregate(avg=models.Avg('pnl'))['avg'] or 0
        risk_reward = abs(float(avg_win) / float(avg_loss)) if avg_loss != 0 else 0
        
        # Group trades by strategy
        strategies = {}
        for trade in trades:
            strategy = trade.strategy or 'Uncategorized'
            if strategy not in strategies:
                strategies[strategy] = {'count': 0, 'pnl': 0}
            strategies[strategy]['count'] += 1
            strategies[strategy]['pnl'] += float(trade.pnl or 0)
        
        # Group trades by asset
        assets = {}
        for trade in trades:
            asset = trade.asset
            if asset not in assets:
                assets[asset] = {'count': 0, 'pnl': 0}
            assets[asset]['count'] += 1
            assets[asset]['pnl'] += float(trade.pnl or 0)
        
        return JsonResponse({
            'success': True,
            'trading_days': trading_days_data,
            'analytics': {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'risk_reward_ratio': risk_reward,
                'percentage_to_target': account.percentage_to_target(),
                'days_remaining': account.days_remaining(),
            },
            'strategies': strategies,
            'assets': assets
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@csrf_exempt
@require_http_methods(["GET"])
def get_prop_metrics(request):
    try:
        metrics, created = ManagementMetrics.objects.get_or_create(id=1)
        
        return JsonResponse({
            'success': True,
            'metrics': {
                'total_accounts': metrics.total_accounts,
                'total_capital_managed': float(metrics.total_capital_managed),
                'total_profit': float(metrics.total_profit),
                'win_rate': float(metrics.win_rate),
                'avg_risk_reward': float(metrics.avg_risk_reward),
                'avg_session_time': metrics.avg_session_time,
            }
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


def update_prop_metrics():
    """Helper function to update aggregate metrics"""
    from django.db import models
    
    accounts = PropFirmAccount.objects.all()
    
    # Calculate metrics
    total_accounts = accounts.count()
    total_capital = accounts.aggregate(sum=models.Sum('current_balance'))['sum'] or 0
    
    # Calculate profit
    total_profit = 0
    for account in accounts:
        profit = account.current_balance - account.initial_balance
        total_profit += profit
    
    # Calculate win rate and risk/reward
    all_trades = PropTrade.objects.filter(exit_time__isnull=False)
    total_trades = all_trades.count()
    winning_trades = all_trades.filter(pnl__gt=0).count()
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    avg_win = all_trades.filter(pnl__gt=0).aggregate(avg=models.Avg('pnl'))['avg'] or 0
    avg_loss = all_trades.filter(pnl__lt=0).aggregate(avg=models.Avg('pnl'))['avg'] or 0
    risk_reward = abs(float(avg_win) / float(avg_loss)) if avg_loss and avg_loss != 0 else 0
    
    # Calculate average session time
    avg_session_time = TradingDay.objects.all().aggregate(avg=models.Avg('session_time_minutes'))['avg'] or 0
    
    # Update or create metrics object
    metrics, created = ManagementMetrics.objects.get_or_create(id=1)
    metrics.total_accounts = total_accounts
    metrics.total_capital_managed = total_capital
    metrics.total_profit = total_profit
    metrics.win_rate = win_rate
    metrics.avg_risk_reward = risk_reward
    metrics.avg_session_time = avg_session_time
    metrics.save()


# Add this function to serve media files in development
def serve_prop_firm_logo(request, path):
    from django.http import FileResponse
    from django.conf import settings
    import os
    
    file_path = os.path.join(settings.MEDIA_ROOT, 'prop_firm_logos', path)
    if os.path.exists(file_path):
        return FileResponse(open(file_path, 'rb'))
    else:
        return JsonResponse({'error': 'File not found'}, status=404)


# views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .models import PropFirm, PropFirmManagementMetrics
from django.core.exceptions import ObjectDoesNotExist

@csrf_exempt
def prop_firm_list(request):
    if request.method == 'GET':
        firms = PropFirm.objects.all()
        return JsonResponse([{
            'id': firm.id,
            'name': firm.name,
            'logo': firm.logo,
            'website': firm.website
        } for firm in firms], safe=False)
    
    elif request.method == 'POST':
        try:
            data = json.loads(request.body)
            new_firm = PropFirm.objects.create(
                name=data.get('name'),
                logo=data.get('logo', ''),
                website=data.get('website', '')
            )
            return JsonResponse({
                'id': new_firm.id,
                'name': new_firm.name,
                'logo': new_firm.logo,
                'website': new_firm.website
            }, status=201)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

@csrf_exempt
def prop_firm_detail(request, firm_id):
    try:
        firm = PropFirm.objects.get(id=firm_id)
    except ObjectDoesNotExist:
        return JsonResponse({'error': 'Prop firm not found'}, status=404)
    
    if request.method == 'GET':
        return JsonResponse({
            'id': firm.id,
            'name': firm.name,
            'logo': firm.logo,
            'website': firm.website
        })
    
    elif request.method == 'PUT':
        try:
            data = json.loads(request.body)
            firm.name = data.get('name', firm.name)
            firm.logo = data.get('logo', firm.logo)
            firm.website = data.get('website', firm.website)
            firm.save()
            return JsonResponse({
                'id': firm.id,
                'name': firm.name,
                'logo': firm.logo,
                'website': firm.website
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    
    elif request.method == 'DELETE':
        firm.delete()
        return JsonResponse({'message': 'Prop firm deleted successfully'})

@csrf_exempt
def metrics_list(request):
    if request.method == 'GET':
        metrics = PropFirmManagementMetrics.objects.all()
        return JsonResponse([{
            'id': metric.id,
            'prop_firm': {
                'id': metric.prop_firm.id,
                'name': metric.prop_firm.name,
                'logo': metric.prop_firm.logo
            },
            'account_type': metric.account_type,
            'status': metric.status,
            'account_id': metric.account_id,
            'starting_balance': float(metric.starting_balance),
            'current_balance': float(metric.current_balance),
            'current_equity': float(metric.current_equity),
            'profit_target': float(metric.profit_target) if metric.profit_target else None,
            'max_drawdown': float(metric.max_drawdown) if metric.max_drawdown else None,
            'start_date': metric.start_date.isoformat(),
            'notes': metric.notes
        } for metric in metrics], safe=False)
    
    elif request.method == 'POST':
        try:
            data = json.loads(request.body)
            prop_firm = PropFirm.objects.get(id=data.get('prop_firm_id'))
            
            new_metric = PropFirmManagementMetrics.objects.create(
                prop_firm=prop_firm,
                account_type=data.get('account_type'),
                status=data.get('status'),
                account_id=data.get('account_id', ''),
                starting_balance=data.get('starting_balance'),
                current_balance=data.get('current_balance'),
                current_equity=data.get('current_equity'),
                profit_target=data.get('profit_target', None),
                max_drawdown=data.get('max_drawdown', None),
                start_date=data.get('start_date'),
                notes=data.get('notes', '')
            )
            
            return JsonResponse({
                'id': new_metric.id,
                'prop_firm': {
                    'id': new_metric.prop_firm.id,
                    'name': new_metric.prop_firm.name
                },
                'account_type': new_metric.account_type,
                'status': new_metric.status
            }, status=201)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

@csrf_exempt
def metric_detail(request, metric_id):
    try:
        metric = PropFirmManagementMetrics.objects.get(id=metric_id)
    except ObjectDoesNotExist:
        return JsonResponse({'error': 'Metric not found'}, status=404)
    
    if request.method == 'GET':
        return JsonResponse({
            'id': metric.id,
            'prop_firm': {
                'id': metric.prop_firm.id,
                'name': metric.prop_firm.name,
                'logo': metric.prop_firm.logo
            },
            'account_type': metric.account_type,
            'status': metric.status,
            'account_id': metric.account_id,
            'starting_balance': float(metric.starting_balance),
            'current_balance': float(metric.current_balance),
            'current_equity': float(metric.current_equity),
            'profit_target': float(metric.profit_target) if metric.profit_target else None,
            'max_drawdown': float(metric.max_drawdown) if metric.max_drawdown else None,
            'start_date': metric.start_date.isoformat(),
            'notes': metric.notes
        })
    
    elif request.method == 'PUT':
        try:
            data = json.loads(request.body)
            
            if 'prop_firm_id' in data:
                metric.prop_firm = PropFirm.objects.get(id=data.get('prop_firm_id'))
            
            fields = ['account_type', 'status', 'account_id', 'starting_balance', 
                     'current_balance', 'current_equity', 'profit_target', 
                     'max_drawdown', 'start_date', 'notes']
            
            for field in fields:
                if field in data:
                    setattr(metric, field, data.get(field))
            
            metric.save()
            
            return JsonResponse({
                'id': metric.id,
                'prop_firm': {
                    'id': metric.prop_firm.id,
                    'name': metric.prop_firm.name
                },
                'status': metric.status,
                'current_balance': float(metric.current_balance),
                'current_equity': float(metric.current_equity)
            })
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    
    elif request.method == 'DELETE':
        metric.delete()
        return JsonResponse({'message': 'Metric deleted successfully'})


from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.shortcuts import get_object_or_404
import json
from datetime import datetime
from .models import EconomicEvent

# Helper function to serialize EconomicEvent objects to dictionaries
def event_to_dict(event):
    return {
        'id': event.id,
        'date_time': event.date_time.isoformat() if hasattr(event.date_time, 'isoformat') else event.date_time,
        'currency': event.currency,
        'impact': event.impact,
        'event_name': event.event_name,
        'actual': event.actual,
        'forecast': event.forecast,
        'previous': event.previous,
        'created_at': event.created_at.isoformat() if hasattr(event.created_at, 'isoformat') else event.created_at,
        'updated_at': event.updated_at.isoformat() if hasattr(event.updated_at, 'isoformat') else event.updated_at
    }

@csrf_exempt
def economic_events_list(request):
    """List all economic events or create a new one"""
    if request.method == 'GET':
        # Filter by date range if provided
        start_date = request.GET.get('start_date')
        end_date = request.GET.get('end_date')
        
        events = EconomicEvent.objects.all()
        
        if start_date and end_date:
            # Format: YYYY-MM-DD
            start_datetime = datetime.strptime(f"{start_date} 00:00:00", "%Y-%m-%d %H:%M:%S")
            end_datetime = datetime.strptime(f"{end_date} 23:59:59", "%Y-%m-%d %H:%M:%S")
            events = events.filter(date_time__range=(start_datetime, end_datetime))
        
        events_data = [event_to_dict(event) for event in events]
        return JsonResponse(events_data, safe=False)
    
    elif request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            # Create a new event
            event = EconomicEvent(
                date_time=data['date_time'],
                currency=data['currency'],
                impact=data['impact'],
                event_name=data['event_name'],
                actual=data.get('actual'),
                forecast=data.get('forecast'),
                previous=data.get('previous')
            )
            event.save()
            
            return JsonResponse(event_to_dict(event), status=201)
        except (KeyError, json.JSONDecodeError):
            return JsonResponse({'error': 'Invalid data provided'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

@csrf_exempt
def economic_event_detail(request, pk):
    """Retrieve, update or delete an economic event"""
    event = get_object_or_404(EconomicEvent, pk=pk)
    
    if request.method == 'GET':
        return JsonResponse(event_to_dict(event))
    
    elif request.method == 'PUT':
        try:
            data = json.loads(request.body)
            
            # Update fields
            if 'date_time' in data:
                event.date_time = data['date_time']
            if 'currency' in data:
                event.currency = data['currency']
            if 'impact' in data:
                event.impact = data['impact']
            if 'event_name' in data:
                event.event_name = data['event_name']
            if 'actual' in data:
                event.actual = data['actual']
            if 'forecast' in data:
                event.forecast = data['forecast']
            if 'previous' in data:
                event.previous = data['previous']
            
            event.save()
            return JsonResponse(event_to_dict(event))
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid data provided'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    
    elif request.method == 'DELETE':
        event.delete()
        return JsonResponse({}, status=204)



# views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.shortcuts import get_object_or_404
import json
from .models import EconomicEvent

@csrf_exempt
@require_http_methods(["GET"])
def data_calendar_economic_events_list(request):
    # Get query parameters
    currency = request.GET.get('currency', '')
    impact = request.GET.get('impact', '')
    
    # Start with all events
    events = EconomicEvent.objects.all()
    
    # Apply filters if provided
    if currency:
        events = events.filter(currency=currency)
    
    if impact:
        events = events.filter(impact=impact)
    
    # Convert to list of dictionaries
    events_data = []
    for event in events:
        events_data.append({
            'id': event.id,
            'date_time': event.date_time.isoformat(),
            'currency': event.currency,
            'impact': event.impact,
            'event_name': event.event_name,
            'actual': event.actual,
            'forecast': event.forecast,
            'previous': event.previous,
        })
    
    return JsonResponse(events_data, safe=False)

@csrf_exempt
@require_http_methods(["GET"])
def data_calendar_economic_event_detail(request, event_id):
    event = get_object_or_404(EconomicEvent, id=event_id)
    
    event_data = {
        'id': event.id,
        'date_time': event.date_time.isoformat(),
        'currency': event.currency,
        'impact': event.impact,
        'event_name': event.event_name,
        'actual': event.actual,
        'forecast': event.forecast,
        'previous': event.previous,
        'created_at': event.created_at.isoformat(),
        'updated_at': event.updated_at.isoformat(),
    }
    
    return JsonResponse(event_data)


@csrf_exempt
@require_http_methods(["GET"])
def unique_economic_events_list(request):
    # Get query parameters
    currency = request.GET.get('currency', '')
    impact = request.GET.get('impact', '')
    search_term = request.GET.get('search', '')
    
    # Get filtered QuerySet
    query = EconomicEvent.objects.all()
    
    # Apply filters if provided
    if currency:
        query = query.filter(currency=currency)
    
    if impact:
        query = query.filter(impact=impact)
        
    if search_term:
        query = query.filter(event_name__icontains=search_term)
    
    # Get distinct event names (case insensitive)
    # Use a dictionary to track unique event names with case insensitivity
    unique_events = {}
    
    for event in query:
        # Use lowercase as the key to ensure case insensitivity
        event_key = event.event_name.lower().strip()
        
        # Only add if we haven't seen this event name before
        if event_key not in unique_events:
            unique_events[event_key] = {
                'id': event.id,
                'event_name': event.event_name,
                'currency': event.currency,
                'impact': event.impact,
            }
    
    # Convert the dictionary values to a list
    events_data = list(unique_events.values())
    
    # Sort the results by event_name
    events_data.sort(key=lambda x: x['event_name'])
    
    return JsonResponse(events_data, safe=False)
    

@csrf_exempt
@require_http_methods(["GET"])
def event_history(request, event_name):
    # Get query parameters for additional filtering
    currency = request.GET.get('currency', '')
    impact = request.GET.get('impact', '')
    
    # Initialize query to get all matching events with the same name
    query = EconomicEvent.objects.filter(event_name=event_name)
    
    # Apply additional filters if provided
    if currency:
        query = query.filter(currency=currency)
    
    if impact:
        query = query.filter(impact=impact)
    
    # Order by date_time
    query = query.order_by('date_time')
    
    # Process the data for charts
    events_data = []
    for event in query:
        # Clean the actual and forecast values for chart display
        actual_value = clean_numeric_value(event.actual)
        forecast_value = clean_numeric_value(event.forecast)
        previous_value = clean_numeric_value(event.previous)
        
        events_data.append({
            'id': event.id,
            'date': event.date_time.strftime('%Y-%m-%d'),
            'currency': event.currency,
            'impact': event.impact,
            'event_name': event.event_name,
            'actual': event.actual,  # Original string value
            'forecast': event.forecast,  # Original string value
            'previous': event.previous,  # Original string value
            'actual_value': actual_value,  # Cleaned numeric value for charts
            'forecast_value': forecast_value,  # Cleaned numeric value for charts
            'previous_value': previous_value,  # Cleaned numeric value for charts
        })
    
    return JsonResponse({
        'event_name': event_name,
        'currency': currency if currency else 'All',
        'impact': impact if impact else 'All',
        'data_points': len(events_data),
        'history': events_data
    })

def clean_numeric_value(value_str):
    """
    Convert string values like '3.2%', '$50.4B', etc. to float values
    for charting purposes
    """
    if not value_str or value_str.strip() == '':
        return None
    
    try:
        # Remove common symbols and convert to float
        cleaned = value_str.replace('%', '').replace('$', '')
        cleaned = cleaned.replace('K', '').replace('M', '').replace('B', '').replace('T', '')
        cleaned = cleaned.replace(',', '')
        return float(cleaned)
    except (ValueError, TypeError):
        return None

@csrf_exempt
def generate_econ_ai_summary(request):
    """Generate an AI summary based on COT data and economic events with improved styling."""
    try:
        if request.method != 'POST':
            return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)
        
        data = json.loads(request.body)
        prompt = data.get('prompt', '')
        api_key = data.get('api_key', '')
        currency_code = data.get('currency_code', '')
        
        if not prompt:
            return JsonResponse({'error': 'Prompt is required'}, status=400)
        
        if not api_key:
            return JsonResponse({'error': 'API key is required'}, status=400)
            
        if not currency_code:
            return JsonResponse({'error': 'Currency code is required'}, status=400)
        
        # Get economic events for the currency
        economic_events = get_economic_events_for_currency(currency_code)
        
        # Append economic events data to the prompt
        prompt_with_events = prompt + "\n\n" + economic_events
        
        # Enhanced system prompt for more engaging responses
        system_prompt = """You are a financial analyst specializing in economic data and market analysis. 
        Provide concise, insightful analyses of economic data and recent economic events.
        
        Follow these style guidelines to make your response more visually appealing:
        1. Use appropriate emojis to highlight key points (1-2 emojis per section, don't overuse)
        2. Add clear section headers with emojis (e.g., "📊 Market Positioning")
        3. Use bullet points for key takeaways
        4. Include a "Bottom Line" summary at the end
        5. Ensure the content is well-organized and easy to scan
        6. Keep the overall analysis professional but engaging
        7. Bold important terms or conclusions
        8. DO NOT USE MARKDOWN FORMATTING
        
        Sections to include:
        - 📊 Current Positioning Analysis
        - 🔮 Economic Outlook
        - 📅 Recent Events Impact
        - 💹 Market Implications
        - 💡 Bottom Line
        """
        
        # Set the API key
        openai.api_key = api_key
        
        # Generate summary using GPT-4o-mini with enhanced prompt
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_with_events}
            ],
            max_tokens=600,  # Slightly increased to accommodate formatting
            temperature=0.7
        )
        
        summary = response.choices[0].message.content.replace('**', '')
        
        return JsonResponse({'summary': summary})
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

# def get_economic_events_for_currency(currency_code):
#     """Get recent economic events for a specified currency."""
#     try:
#         # Get events from the last 30 days
#         thirty_days_ago = timezone.now() - timezone.timedelta(days=90)
#         events = EconomicEvent.objects.filter(
#             currency=currency_code,
#             date_time__gte=thirty_days_ago
#         ).order_by('-date_time')
        
#         if not events:
#             return "No recent economic events found for this currency."
        
#         # Format the events data
#         events_text = "Recent Economic Events for this currency:\n\n"
#         for event in events:
#             impact_symbol = "🔴" if event.impact == "high" else "🟠" if event.impact == "medium" else "🟢"
#             events_text += f"Date: {event.date_time.strftime('%Y-%m-%d %H:%M')}\n"
#             events_text += f"Event: {event.event_name} {impact_symbol}\n"
#             events_text += f"Actual: {event.actual or 'N/A'}\n"
#             events_text += f"Forecast: {event.forecast or 'N/A'}\n"
#             events_text += f"Previous: {event.previous or 'N/A'}\n\n"
        
#         return events_text
    
#     except Exception as e:
#         return f"Error retrieving economic events: {str(e)}"


@csrf_exempt
def generate_econ_cot_data(request):
    try:
        # Get requested assets from POST data if provided
        if request.method == 'POST':
            requested_assets = json.loads(request.body).get('assets', [])
        else:
            # Default assets for GET requests
            requested_assets = [
                'USD INDEX - ICE FUTURES U.S.',
                'EURO FX - CHICAGO MERCANTILE EXCHANGE',
                'BRITISH POUND - CHICAGO MERCANTILE EXCHANGE',
                'CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE',
                'SWISS FRANC - CHICAGO MERCANTILE EXCHANGE',
                'JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE',
                'NZ DOLLAR - CHICAGO MERCANTILE EXCHANGE',
                'AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE',
                'GOLD - COMMODITY EXCHANGE INC.',
                'UST BOND - CHICAGO BOARD OF TRADE',
                'UST 10Y NOTE - CHICAGO BOARD OF TRADE',
                'UST 5Y NOTE - CHICAGO BOARD OF TRADE',
                'NASDAQ MINI - CHICAGO MERCANTILE EXCHANGE',
                'E-MINI S&P 500 -',
                'DOW JONES U.S. REAL ESTATE IDX - CHICAGO BOARD OF TRADE'
            ]

        # Get the current year and previous year
        current_year = pd.Timestamp.now().year
        previous_year = current_year - 1

        # Create list to store DataFrames
        df_list = []

        # Fetch data for previous and current year
        for year in range(previous_year, current_year + 1):
            single_year = cot.cot_year(year, cot_report_type='legacy_futopt')
            df_list.append(single_year)

        # Concatenate all DataFrames
        df = pd.concat(df_list, ignore_index=True)

        # Convert dates to datetime
        df['As of Date in Form YYYY-MM-DD'] = pd.to_datetime(df['As of Date in Form YYYY-MM-DD'])

        # Filter for current year data
        unfiltered_currency_df = df[df['As of Date in Form YYYY-MM-DD'].dt.year == current_year]

        # Filter for requested assets
        unfiltered_currency_df = unfiltered_currency_df[
            unfiltered_currency_df['Market and Exchange Names'].isin(requested_assets)
        ]

        # Remove specific exclusions (e.g., MICRO GOLD)
        unfiltered_currency_df = unfiltered_currency_df[
            unfiltered_currency_df['Market and Exchange Names'] != 'MICRO GOLD - COMMODITY EXCHANGE INC.'
        ]

        # Fill missing values and ensure numeric columns
        numeric_columns = [
            'Noncommercial Positions-Long (All)',
            'Noncommercial Positions-Short (All)',
            'Commercial Positions-Long (All)',
            'Commercial Positions-Short (All)'
        ]
        
        unfiltered_currency_df[numeric_columns] = unfiltered_currency_df[numeric_columns].fillna(0).astype(float)

        # Calculate net positions for unfiltered data
        unfiltered_currency_df['Net Noncommercial Positions'] = (
            unfiltered_currency_df['Noncommercial Positions-Long (All)'] - 
            unfiltered_currency_df['Noncommercial Positions-Short (All)']
        )
        unfiltered_currency_df['Net Commercial Positions'] = (
            unfiltered_currency_df['Commercial Positions-Long (All)'] - 
            unfiltered_currency_df['Commercial Positions-Short (All)']
        )

        # Get the rows with maximum open interest for each market
        idx = unfiltered_currency_df.groupby('Market and Exchange Names')['Open Interest (All)'].idxmax()
        currency_df = unfiltered_currency_df.loc[idx]

        # Calculate total positions
        currency_df['Total Noncommercial Positions'] = (
            currency_df['Noncommercial Positions-Long (All)'] + 
            currency_df['Noncommercial Positions-Short (All)']
        )
        currency_df['Total Commercial Positions'] = (
            currency_df['Commercial Positions-Long (All)'] + 
            currency_df['Commercial Positions-Short (All)']
        )
        currency_df['Total Positions'] = (
            currency_df['Total Noncommercial Positions'] + 
            currency_df['Total Commercial Positions']
        )

        # Calculate percentages
        currency_df['Percentage Noncommercial Long'] = (
            currency_df['Noncommercial Positions-Long (All)'] / 
            currency_df['Total Noncommercial Positions']
        ) * 100
        currency_df['Percentage Noncommercial Short'] = (
            currency_df['Noncommercial Positions-Short (All)'] / 
            currency_df['Total Noncommercial Positions']
        ) * 100
        currency_df['Percentage Commercial Long'] = (
            currency_df['Commercial Positions-Long (All)'] / 
            currency_df['Total Commercial Positions']
        ) * 100
        currency_df['Percentage Commercial Short'] = (
            currency_df['Commercial Positions-Short (All)'] / 
            currency_df['Total Commercial Positions']
        ) * 100

        # Generate plots
        plot_urls = plot_net_positions(unfiltered_currency_df)

        # Prepare response data
        data = {}
        round_off_number = 2

        for asset in requested_assets:
            asset_df = currency_df[currency_df['Market and Exchange Names'] == asset]
            
            if not asset_df.empty:
                latest_data = asset_df.iloc[0]
                data[asset] = {
                    'Date': latest_data['As of Date in Form YYYY-MM-DD'].strftime('%Y-%m-%d'),
                    'Percentage_Noncommercial_Long': round(latest_data['Percentage Noncommercial Long'], round_off_number),
                    'Percentage_Noncommercial_Short': round(latest_data['Percentage Noncommercial Short'], round_off_number),
                    'Percentage_Commercial_Long': round(latest_data['Percentage Commercial Long'], round_off_number),
                    'Percentage_Commercial_Short': round(latest_data['Percentage Commercial Short'], round_off_number),
                    'Plot_URL': plot_urls.get(asset, '')
                }

        return JsonResponse(data)
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

logger = logging.getLogger(__name__)

# @csrf_exempt
# @require_http_methods(["POST"])
# def save_forex_factory_news(request):
#     try:
#         # Parse JSON data from request body
#         data = json.loads(request.body)
#         events_data = data.get('events', [])
        
#         if not events_data:
#             return JsonResponse({
#                 'success': False,
#                 'error': 'No events data provided'
#             }, status=400)
        
#         saved_events = []
#         errors = []
        
#         for i, event_data in enumerate(events_data):
#             try:
#                 # Validate required fields
#                 required_fields = ['date_time', 'currency', 'impact', 'event_name']
#                 missing_fields = [field for field in required_fields if not event_data.get(field)]
                
#                 if missing_fields:
#                     errors.append(f"Event {i+1}: Missing required fields: {', '.join(missing_fields)}")
#                     continue
                
#                 # Parse and validate datetime
#                 date_time_str = event_data.get('date_time')
#                 try:
#                     parsed_datetime = parse_datetime(date_time_str)
#                     if not parsed_datetime:
#                         # Try alternative format if the first parsing fails
#                         from datetime import datetime
#                         parsed_datetime = datetime.fromisoformat(date_time_str.replace('Z', '+00:00'))
#                 except (ValueError, TypeError) as e:
#                     errors.append(f"Event {i+1}: Invalid datetime format: {date_time_str}")
#                     continue
                
#                 # Validate currency
#                 valid_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'CNY']
#                 currency = event_data.get('currency', '').upper()
#                 if currency not in valid_currencies:
#                     errors.append(f"Event {i+1}: Invalid currency: {currency}")
#                     continue
                
#                 # Validate impact
#                 valid_impacts = ['low', 'medium', 'high']
#                 impact = event_data.get('impact', '').lower()
#                 if impact not in valid_impacts:
#                     errors.append(f"Event {i+1}: Invalid impact level: {impact}")
#                     continue
                
#                 # Prepare event data for saving
#                 event_kwargs = {
#                     'date_time': parsed_datetime,
#                     'currency': currency,
#                     'impact': impact,
#                     'event_name': event_data.get('event_name', '').strip(),
#                     'actual': event_data.get('actual') or None,
#                     'forecast': event_data.get('forecast') or None,
#                     'previous': event_data.get('previous') or None,
#                 }
                
#                 # Check for duplicates (same datetime, currency, and event_name)
#                 existing_event = EconomicEvent.objects.filter(
#                     date_time=parsed_datetime,
#                     currency=currency,
#                     event_name=event_kwargs['event_name']
#                 ).first()
                
#                 if existing_event:
#                     # Update existing event
#                     for key, value in event_kwargs.items():
#                         setattr(existing_event, key, value)
#                     existing_event.save()
#                     saved_events.append({
#                         'id': existing_event.id,
#                         'action': 'updated',
#                         'event_name': existing_event.event_name
#                     })
#                     logger.info(f"Updated existing event: {existing_event}")
#                 else:
#                     # Create new event
#                     new_event = EconomicEvent.objects.create(**event_kwargs)
#                     saved_events.append({
#                         'id': new_event.id,
#                         'action': 'created',
#                         'event_name': new_event.event_name
#                     })
#                     logger.info(f"Created new event: {new_event}")
                    
#             except Exception as e:
#                 logger.error(f"Error processing event {i+1}: {str(e)}")
#                 errors.append(f"Event {i+1}: {str(e)}")
#                 continue
        
#         # Prepare response
#         response_data = {
#             'success': True,
#             'saved_count': len(saved_events),
#             'error_count': len(errors),
#             'saved_events': saved_events
#         }
        
#         if errors:
#             response_data['errors'] = errors
#             response_data['message'] = f"Saved {len(saved_events)} events with {len(errors)} errors"
#         else:
#             response_data['message'] = f"Successfully saved {len(saved_events)} events"
        
#         # Return appropriate status code
#         status_code = 200 if saved_events else 400
        
#         return JsonResponse(response_data, status=status_code)
        
#     except json.JSONDecodeError:
#         logger.error("Invalid JSON data received")
#         return JsonResponse({
#             'success': False,
#             'error': 'Invalid JSON data'
#         }, status=400)
        
#     except Exception as e:
#         logger.error(f"Unexpected error in save_forex_factory_news: {str(e)}")
#         return JsonResponse({
#             'success': False,
#             'error': 'Internal server error'
#         }, status=500)


# Optional: Add a view to retrieve saved events for verification
# @csrf_exempt
# @require_http_methods(["GET"])
# def get_forex_factory_events(request):
#     """
#     Retrieve economic events with optional filtering
#     """
#     try:
#         # Get query parameters for filtering
#         currency = request.GET.get('currency')
#         impact = request.GET.get('impact')
#         start_date = request.GET.get('start_date')
#         end_date = request.GET.get('end_date')
#         limit = int(request.GET.get('limit', 100))  # Default limit of 100
        
#         # Start with all events
#         queryset = EconomicEvent.objects.all()
        
#         # Apply filters
#         if currency:
#             queryset = queryset.filter(currency=currency.upper())
        
#         if impact:
#             queryset = queryset.filter(impact=impact.lower())
            
#         if start_date:
#             try:
#                 start_dt = parse_datetime(start_date)
#                 if start_dt:
#                     queryset = queryset.filter(date_time__gte=start_dt)
#             except ValueError:
#                 pass
                
#         if end_date:
#             try:
#                 end_dt = parse_datetime(end_date)
#                 if end_dt:
#                     queryset = queryset.filter(date_time__lte=end_dt)
#             except ValueError:
#                 pass
        
#         # Limit results and order by date
#         events = queryset.order_by('-date_time')[:limit]
        
#         # Serialize events
#         events_data = []
#         for event in events:
#             events_data.append({
#                 'id': event.id,
#                 'date_time': event.date_time.isoformat(),
#                 'currency': event.currency,
#                 'impact': event.impact,
#                 'event_name': event.event_name,
#                 'actual': event.actual,
#                 'forecast': event.forecast,
#                 'previous': event.previous,
#                 'created_at': event.created_at.isoformat(),
#                 'updated_at': event.updated_at.isoformat(),
#             })
        
#         return JsonResponse({
#             'success': True,
#             'events': events_data,
#             'count': len(events_data)
#         })
        
#     except Exception as e:
#         logger.error(f"Error retrieving events: {str(e)}")
#         return JsonResponse({
#             'success': False,
#             'error': 'Failed to retrieve events'
#         }, status=500)


# @csrf_exempt
# @require_http_methods(["POST"])
# def process_forex_screenshot(request):
#     """
#     Process Forex Factory screenshot using OpenAI GPT-4o-mini
#     """
#     try:
#         data = json.loads(request.body)
#         image_base64 = data.get('image')
#         api_key = data.get('api_key')
        
#         if not image_base64 or not api_key:
#             return JsonResponse({
#                 'error': 'Missing image or API key'
#             }, status=400)
        
#         # Initialize OpenAI client with the new syntax
#         client = OpenAI(api_key=api_key)
        
#         # Prepare the prompt for GPT-4o-mini
#         prompt = """
#         Analyze this Forex Factory economic calendar screenshot and extract all economic events.
#         Return a JSON response with an 'events' array containing objects with these fields:
#         - date_time: ISO format datetime (YYYY-MM-DDTHH:MM)
#         - currency: 3-letter currency code (USD, EUR, GBP, JPY, AUD, CAD, CHF, CNY)
#         - impact: 'low', 'medium', or 'high'
#         - event_name: Name of the economic event
#         - actual: Actual value (can be empty string if not available)
#         - forecast: Forecasted value (can be empty string if not available)  
#         - previous: Previous value (can be empty string if not available)
        
#         Example response format:
#         {
#           "events": [
#             {
#               "date_time": "2024-12-01T10:00",
#               "currency": "USD",
#               "impact": "high",
#               "event_name": "Non-Farm Payrolls",
#               "actual": "227K",
#               "forecast": "220K",
#               "previous": "12K"
#             }
#           ]
#         }
        
#         Extract ALL visible events from the screenshot. Pay attention to:
#         - Time stamps and convert to 24-hour format
#         - Currency flags/symbols
#         - Impact levels (usually shown as colored indicators)
#         - Event names
#         - Actual, forecast, and previous values
        
#         Return only valid JSON without any additional text or formatting.
#         """
        
#         # Make API call to OpenAI using the new client syntax
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {
#                     "role": "user",
#                     "content": [
#                         {
#                             "type": "text",
#                             "text": prompt
#                         },
#                         {
#                             "type": "image_url",
#                             "image_url": {
#                                 "url": f"data:image/jpeg;base64,{image_base64}"
#                             }
#                         }
#                     ]
#                 }
#             ],
#             max_tokens=2000,
#             temperature=0.1
#         )
        
#         # Extract and parse the response
#         gpt_response = response.choices[0].message.content.strip()
        
#         # Try to parse JSON response
#         try:
#             parsed_data = json.loads(gpt_response)
#             events = parsed_data.get('events', [])
            
#             # Validate and clean the events data
#             cleaned_events = []
#             for event in events:
#                 try:
#                     # Validate required fields
#                     if not all(field in event for field in ['date_time', 'currency', 'impact', 'event_name']):
#                         continue
                    
#                     # Validate currency
#                     valid_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'CNY']
#                     if event['currency'] not in valid_currencies:
#                         event['currency'] = 'USD'  # Default fallback
                    
#                     # Validate impact
#                     valid_impacts = ['low', 'medium', 'high']
#                     if event['impact'] not in valid_impacts:
#                         event['impact'] = 'medium'  # Default fallback
                    
#                     # Ensure optional fields exist
#                     event['actual'] = event.get('actual', '')
#                     event['forecast'] = event.get('forecast', '')
#                     event['previous'] = event.get('previous', '')
                    
#                     # Validate datetime format
#                     try:
#                         datetime.fromisoformat(event['date_time'].replace('Z', '+00:00'))
#                     except ValueError:
#                         # If invalid datetime, skip this event
#                         continue
                    
#                     cleaned_events.append(event)
                    
#                 except Exception as e:
#                     # Skip invalid events
#                     continue
            
#             return JsonResponse({
#                 'success': True,
#                 'events': cleaned_events,
#                 'message': f'Successfully extracted {len(cleaned_events)} events'
#             })
            
#         except json.JSONDecodeError:
#             return JsonResponse({
#                 'error': 'Failed to parse GPT response as JSON',
#                 'gpt_response': gpt_response
#             }, status=500)
            
#     except Exception as e:
#         # More specific error handling
#         print(f'Error occured in process_fx function: {e}')
#         if "OpenAI" in str(type(e)):
#             return JsonResponse({
#                 'error': f'OpenAI API error: {str(e)}'
#             }, status=500)
#         else:
#             return JsonResponse({
#                 'error': f'Server error: {str(e)}'
#             }, status=500)



logger = logging.getLogger(__name__)

@csrf_exempt
@require_http_methods(["POST"])
def save_forex_factory_news(request):
    """
    Save economic events from forex factory screenshot analysis
    Expected JSON format:
    {
        "events": [
            {
                "date_time": "2024-12-01T14:30:00",
                "currency": "USD",
                "impact": "high",
                "event_name": "Non-Farm Payrolls",
                "actual": "150K",
                "forecast": "160K",
                "previous": "140K"
            }
        ]
    }
    """
    try:
        # Parse JSON data
        data = json.loads(request.body)
        events_data = data.get('events', [])
        
        if not events_data:
            return JsonResponse({
                'success': False,
                'error': 'No events provided'
            }, status=400)
        
        # Validate and save events
        saved_events = []
        errors = []
        
        for i, event_data in enumerate(events_data):
            try:
                # Validate required fields
                required_fields = ['date_time', 'currency', 'impact', 'event_name']
                for field in required_fields:
                    if not event_data.get(field):
                        raise ValidationError(f"Missing required field: {field}")
                
                # Parse datetime
                try:
                    if isinstance(event_data['date_time'], str):
                        # Handle different datetime formats
                        dt_str = event_data['date_time']
                        if 'T' in dt_str:
                            if dt_str.endswith('Z'):
                                dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                            else:
                                dt = datetime.fromisoformat(dt_str)
                        else:
                            dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
                    else:
                        dt = event_data['date_time']
                except (ValueError, TypeError) as e:
                    raise ValidationError(f"Invalid date_time format: {e}")
                
                # Validate currency
                valid_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'CNY']
                if event_data['currency'] not in valid_currencies:
                    raise ValidationError(f"Invalid currency: {event_data['currency']}")
                
                # Validate impact
                valid_impacts = ['low', 'medium', 'high']
                if event_data['impact'] not in valid_impacts:
                    raise ValidationError(f"Invalid impact: {event_data['impact']}")
                
                # Create or update event
                # Check if event already exists (same datetime, currency, event_name)
                existing_event = EconomicEvent.objects.filter(
                    date_time=dt,
                    currency=event_data['currency'],
                    event_name=event_data['event_name']
                ).first()
                
                if existing_event:
                    # Update existing event
                    existing_event.impact = event_data['impact']
                    existing_event.actual = event_data.get('actual', '') or ''
                    existing_event.forecast = event_data.get('forecast', '') or ''
                    existing_event.previous = event_data.get('previous', '') or ''
                    existing_event.save()
                    saved_events.append({
                        'id': existing_event.id,
                        'action': 'updated',
                        'event_name': existing_event.event_name
                    })
                else:
                    # Create new event
                    new_event = EconomicEvent.objects.create(
                        date_time=dt,
                        currency=event_data['currency'],
                        impact=event_data['impact'],
                        event_name=event_data['event_name'],
                        actual=event_data.get('actual', '') or '',
                        forecast=event_data.get('forecast', '') or '',
                        previous=event_data.get('previous', '') or ''
                    )
                    saved_events.append({
                        'id': new_event.id,
                        'action': 'created',
                        'event_name': new_event.event_name
                    })
                
            except ValidationError as e:
                errors.append(f"Event {i+1}: {str(e)}")
                logger.error(f"Validation error for event {i+1}: {e}")
            except Exception as e:
                errors.append(f"Event {i+1}: Unexpected error - {str(e)}")
                logger.error(f"Unexpected error for event {i+1}: {e}")
        
        # Prepare response
        response_data = {
            'success': True,
            'saved_count': len(saved_events),
            'total_count': len(events_data),
            'saved_events': saved_events
        }
        
        if errors:
            response_data['errors'] = errors
            response_data['error_count'] = len(errors)
        
        status_code = 200 if saved_events else 400
        return JsonResponse(response_data, status=status_code)
        
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON format'
        }, status=400)
    
    except Exception as e:
        logger.error(f"Unexpected error in save_forex_factory_news: {e}")
        return JsonResponse({
            'success': False,
            'error': 'Internal server error'
        }, status=500)


@csrf_exempt
# create views here


# LEGODI BACKEND CODE
def send_simple_message():
    # Replace with your Mailgun domain and API key
    domain = os.environ['MAILGUN_DOMAIN']
    api_key = os.environ['MAILGUN_API_KEY']

    # Mailgun API endpoint for sending messages
    url = f"https://api.mailgun.net/v3/{domain}/messages"

    # Email details
    sender = f"Excited User <postmaster@{domain}>"
    recipients = ["motingwetlotlo@yahoo.com"]
    subject = "Hello from Mailgun"
    text = "Testing some Mailgun awesomeness!"

    # Send the email
    response = requests.post(url, auth=("api", api_key), data={
        "from": sender,
        "to": recipients,
        "subject": subject,
        "text": text
    })

    # Return the response content as a JSON object
    return {
        "status_code": response.status_code,
        "response_content": response.content.decode("utf-8")
    }


def contact_us(request):
    if request.method == "POST":
        # Get form data from request body
        data = json.loads(request.body)
        first_name = data.get("firstName")
        last_name = data.get("lastName")
        email = data.get("email")
        message = data.get("message")
        
        # Save form data to the ContactUs model
        contact_us_entry = ContactUs.objects.create(
            first_name=first_name,
            last_name=last_name,
            email=email,
            message=message
        )
        return JsonResponse({"message": "Email sent successfully and saved to database!"})
    else:
        return JsonResponse({"error": "Method not allowed"}, status=405)


def book_order(request):
    if request.method == "POST":
        # Get form data from request body
        try:
            data = json.loads(request.body)
            first_name = data.get("first_name")
            last_name = data.get("last_name")
            email = data.get("email")
            interested_product = data.get("interested_product")
            number_of_units = int(data.get("number_of_units"))
            phone_number = data.get("phone_number")

            # Save form data to the BookOrder model
            book_order_entry = BookOrder.objects.create(
                first_name=first_name,
                last_name=last_name,
                email=email,
                interested_product=interested_product,
                phone_number=phone_number,
                number_of_units=number_of_units
            )
            return JsonResponse({"message": "Order booked successfully!"})
        except Exception as e:
            print(f'Exception occured: {e}')
            return JsonResponse({'error': str(e)})
    else:
        return JsonResponse({"error": "Method not allowed"}, status=405)

# Legodi Tech Registration and Login
from rest_framework import generics

class UserRegistrationView(generics.CreateAPIView):
    queryset = CustomUser.objects.all()
    serializer_class = CustomUserSerializer


def user_login(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        user = authenticate(request, username=email, password=password)
        if user:
            # User is authenticated
            login(request, user)
            # Generate and return an authentication token (e.g., JWT)
            return JsonResponse({'message': 'Login successful', 'token': 'your_token_here'})
        else:
            return JsonResponse({'message': 'Invalid credentials'}, status=400)
    else:
        return JsonResponse({'message': 'Invalid request method'}, status=405)


from django.middleware.csrf import get_token
from django.http import JsonResponse

def get_csrf_token(request):
    try:
        csrf_token = get_token(request)
        return JsonResponse({'csrfToken': csrf_token})
    except Exception as e:
        return JsonResponse({'error': str(e)})
