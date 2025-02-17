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
from typing import List, Optional, Dict, Any
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


from PIL import Image
import io



# Comment
# current_hour = datetime.datetime.now().time().hour



scheduler = BackgroundScheduler()
scheduler.start()


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


async def genesys_backest(code):

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
                exec(code)    
            except Exception as e:
                print(f'Exception: {e}')
    try:
        # Query the model asynchronously using sync_to_async
        queryset = await sync_to_async(SaveDataset.objects.all().first)()
        dataset_to_use = f'./{queryset.dataset}'
        df_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataset_to_use)
        df = pd.read_csv(df_path).drop_duplicates()
        df.index = pd.to_datetime(df['Time'].values)
        del df['Time']
        
        split_queryset = await sync_to_async(SplitDataset.objects.get)()

        start_year = int(split_queryset.start_year)
        end_year = int(split_queryset.end_year)
        new_df = split_df(df, start_year, end_year)
        # print(df)
        
        init_capital_queryset = await sync_to_async(SetInitCapital.objects.get)()
        init_capital = float(init_capital_queryset.initial_capital)

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
                    result = await genesys_backest(generated_code)
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


def obtain_dataset(asset, interval, num_days):

    # Calculate the date 30 days ago from the current day
    start_date = (datetime.now() - timedelta(days=num_days)).strftime("%Y-%m-%d")

    # Get latest candle
    end_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    # Download data using the calculated dates
    forex_asset = f"{asset}=X"
    data = yf.download(forex_asset, start=start_date, end=end_date, interval=interval)

    return data


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


# @csrf_exempt
# def generate_cot_data(request):
#     try:
#          # Example: cot_hist()
#         df = cot.cot_hist(cot_report_type='traders_in_financial_futures_futopt')
#         # cot_hist() downloads the historical bulk file for the specified report type, in this example the Traders in Financial Futures Futures-and-Options Combined report. Returns the data as dataframe.

#         # Filter for the current year
#         current_year = pd.Timestamp.now().year  # Get the current year
#         previous_year = current_year - 1

#         # Example: cot_year()
#         df = cot.cot_year(year=previous_year, cot_report_type='traders_in_financial_futures_fut')
#         # cot_year() downloads the single year file of the specified report type and year. Returns the data as dataframe.

#         # Example for collecting data of a few years, here from 2017 to 2020, of a specified report:
#         df_list = []  # Create an empty list to hold DataFrames
#         begin_year = previous_year
#         end_year = current_year

#         for i in range(begin_year, end_year + 1):
#             single_year = cot.cot_year(i, cot_report_type='legacy_futopt')
#             df_list.append(single_year)  # Append each DataFrame to the list

#         df = pd.concat(df_list, ignore_index=True)  # Concatenate all DataFrames in the list

#         # Example: cot_all()
#         df = cot.cot_all(cot_report_type='legacy_fut')
#         # cot_all() downloads the historical bulk file and all remaining single year files of the specified report type.  Returns the data as dataframe.

#         # Ensure 'As of Date in Form YYYY-MM-DD' is in datetime format
#         df['As of Date in Form YYYY-MM-DD'] = pd.to_datetime(df['As of Date in Form YYYY-MM-DD'])

#         # Filter the data for the current year
#         currency_df = df[df['As of Date in Form YYYY-MM-DD'].dt.year == current_year]

#         # Define your currency keywords
#         currency_keywords = ['USD INDEX', 'EURO FX - CHICAGO MERCANTILE EXCHANGE', 'BRITISH POUND - CHICAGO MERCANTILE EXCHANGE', 'GOLD - COMMODITY EXCHANGE INC', 'UST 5Y NOTE - CHICAGO BOARD OF TRADE', 'UST 10Y NOTE - CHICAGO BOARD OF TRADE', 'UST BOND - CHICAGO BOARD OF TRADE', 'NASDAQ MINI - CHICAGO MERCANTILE EXCHANGE', 'E-MINI S&P 500 -', 'DOW JONES U.S. REAL ESTATE IDX - CHICAGO BOARD OF TRADE']

#         # Filter the DataFrame for the current year and specific currencies
#         unfiltered_currency_df = df[df['As of Date in Form YYYY-MM-DD'].dt.year == current_year]
#         unfiltered_currency_df = unfiltered_currency_df[unfiltered_currency_df['Market and Exchange Names'].str.contains('|'.join(currency_keywords), case=False, na=False)]
#         unfiltered_currency_df = unfiltered_currency_df[unfiltered_currency_df['Market and Exchange Names'] != 'MICRO GOLD - COMMODITY EXCHANGE INC.']


#         # Group by 'Market and Exchange Names' and get the index of the maximum open interest
#         # idx = currency_df.groupby('Market and Exchange Names')['Open Interest (All)'].idxmax()
#         # currency_df = currency_df.loc[idx]

#         # Fill missing values and ensure columns are numeric
#         unfiltered_currency_df[['Noncommercial Positions-Long (All)', 'Noncommercial Positions-Short (All)',
#                     'Commercial Positions-Long (All)', 'Commercial Positions-Short (All)']] = unfiltered_currency_df[[
#             'Noncommercial Positions-Long (All)', 'Noncommercial Positions-Short (All)',
#             'Commercial Positions-Long (All)', 'Commercial Positions-Short (All)'
#         ]].fillna(0).astype(float)

#         # Calculate net positions
#         unfiltered_currency_df['Net Noncommercial Positions'] = unfiltered_currency_df['Noncommercial Positions-Long (All)'] - unfiltered_currency_df['Noncommercial Positions-Short (All)']
#         unfiltered_currency_df['Net Commercial Positions'] = unfiltered_currency_df['Commercial Positions-Long (All)'] - unfiltered_currency_df['Commercial Positions-Short (All)']

#         # Further filter the DataFrame for rows that match any of the currency keywords
#         currency_df = currency_df[currency_df['Market and Exchange Names'].str.contains('|'.join(currency_keywords), case=False, na=False)]

#         # Group by 'Market and Exchange Names' and get the index of the maximum open interest
#         idx = currency_df.groupby('Market and Exchange Names')['Open Interest (All)'].idxmax()

#         # Filter the DataFrame to include only the rows with the highest open interest for each market
#         currency_df = currency_df.loc[idx]

#         # df = currency_df

#         # Ensure there are no missing values and the columns are numeric
#         currency_df[['Noncommercial Positions-Long (All)', 'Noncommercial Positions-Short (All)',
#             'Commercial Positions-Long (All)', 'Commercial Positions-Short (All)']] = currency_df[[
#                 'Noncommercial Positions-Long (All)', 'Noncommercial Positions-Short (All)',
#                 'Commercial Positions-Long (All)', 'Commercial Positions-Short (All)'
#             ]].fillna(0).astype(float)

#         # Calculate total positions for each row
#         currency_df['Total Noncommercial Positions'] = currency_df['Noncommercial Positions-Long (All)'] + currency_df['Noncommercial Positions-Short (All)']
#         currency_df['Total Commercial Positions'] = currency_df['Commercial Positions-Long (All)'] + currency_df['Commercial Positions-Short (All)']
#         currency_df['Total Positions'] = currency_df['Total Noncommercial Positions'] + currency_df['Total Commercial Positions']

#         # Calculate net positions
#         currency_df['Net Noncommercial Positions'] = currency_df['Noncommercial Positions-Long (All)'] - currency_df['Noncommercial Positions-Short (All)']
#         currency_df['Net Commercial Positions'] = currency_df['Commercial Positions-Long (All)'] - currency_df['Commercial Positions-Short (All)']

#         # Calculate percentages
#         currency_df['Percentage Noncommercial Long'] = (currency_df['Noncommercial Positions-Long (All)'] / currency_df['Total Noncommercial Positions']) * 100
#         currency_df['Percentage Noncommercial Short'] = (currency_df['Noncommercial Positions-Short (All)'] / currency_df['Total Noncommercial Positions']) * 100
#         currency_df['Percentage Commercial Long'] = (currency_df['Commercial Positions-Long (All)'] / currency_df['Total Commercial Positions']) * 100
#         currency_df['Percentage Commercial Short'] = (currency_df['Commercial Positions-Short (All)'] / currency_df['Total Commercial Positions']) * 100

#         # Prepare the plot data
#         plot_urls = plot_net_positions(unfiltered_currency_df)

#         # Extract data for each specific asset
#         assets = ['USD INDEX - ICE FUTURES U.S.', 'EURO FX - CHICAGO MERCANTILE EXCHANGE', 'BRITISH POUND - CHICAGO MERCANTILE EXCHANGE', 'GOLD - COMMODITY EXCHANGE INC.', 'UST BOND - CHICAGO BOARD OF TRADE', 'UST 10Y NOTE - CHICAGO BOARD OF TRADE', 'UST 5Y NOTE - CHICAGO BOARD OF TRADE', 'NASDAQ MINI - CHICAGO MERCANTILE EXCHANGE', 'E-MINI S&P 500 -', 'DOW JONES U.S. REAL ESTATE IDX - CHICAGO BOARD OF TRADE']
#         data = {}

#         round_off_number = 2

#         for asset in assets:
#             asset_df = currency_df[currency_df['Market and Exchange Names'] == asset]
            
#             if not asset_df.empty:
#                 # Get the most recent data
#                 latest_data = asset_df.iloc[0]
#                 data[asset] = {
#                     'Date': latest_data['As of Date in Form YYYY-MM-DD'].strftime('%Y-%m-%d'),
#                     'Percentage Noncommercial Long': round(latest_data['Percentage Noncommercial Long'], round_off_number),
#                     'Percentage Noncommercial Short': round(latest_data['Percentage Noncommercial Short'], round_off_number),
#                     'Percentage Commercial Long': round(latest_data['Percentage Commercial Long'], round_off_number),
#                     'Percentage Commercial Short': round(latest_data['Percentage Commercial Short'], round_off_number),
#                     'Plot URL': plot_urls.get(asset, '')
#                 }

#         return JsonResponse(data)

#     except Exception as e:
#         print(f'Error occurred in generate_cot_data: {e}')
#         return JsonResponse({'message': f'Error occurred in generate_cot_data: {e}'})


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
                'GOLD - COMMODITY EXCHANGE INC.',
                'UST 5Y NOTE - CHICAGO BOARD OF TRADE',
                'UST 10Y NOTE - CHICAGO BOARD OF TRADE',
                'UST BOND - CHICAGO BOARD OF TRADE',
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

            # Get the most recent closing price
            latest_price = round(data["Close"].iloc[-1], 5)

            # Check the condition
            condition_met = (
                (condition == ">" and latest_price > target_price) or
                (condition == "<" and latest_price < target_price)
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


def analyse_image(image_data, news_data):
    try:
        # Getting the base64 string
        base64_image = base64.b64encode(image_data).decode('utf-8')

        api_key = os.environ['OPENAI_API_KEY']

        # Extract discussion prompt if it exists
        discussion_prompt = news_data.get('discussion_prompt', '')

        # Construct a more interactive prompt
        prompt = f"""
        {discussion_prompt}

        Based on the trading chart image, provide an analysis that addresses the above context.
        Your response should be in JSON format with two keys:
        1. 'analysis': A detailed technical analysis that:
           - Directly responds to any previous trader's points if they exist
           - Explains your reasoning for agreeing or disagreeing
           - Points out any overlooked patterns or indicators
           - Considers both technical and fundamental factors
        2. 'recommendation': Either 'buy', 'sell', or 'neutral'

        Make sure to format as valid JSON and avoid line breaks in the text.

        Additional context: {news_data}
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
            "max_tokens": 1000
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


def analyse_image_from_file(image_path, news_data):
    try:
        # Read the image and encode in base64
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        return analyse_image(image_data, news_data)
    except Exception as e:
        print(f"Error in image analysis from file: {e}")


def fetch_news_data(assets, user_email):

    all_news_data = []

    # List of assets to fetch news data for
    assets_to_fetch = assets

    # Establish a connection to the API
    conn = http.client.HTTPSConnection('api.marketaux.com')

    # Define query parameters
    params_template = {
        'api_token': 'xH2KZ1sYqHmNRpfBVfb9C1BbItHMtlRIdZQoRlYw',
        'langauge': 'en',
        'limit': 1,
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

    return ({'message': all_news_data})

def tradergpt(asset, interval, num_days, user_email):
    try:
        # Step 1: Fetch dataset and generate chart
        data = obtain_dataset(asset, interval, num_days)
        chart_path = generate_candlestick_chart(data)

        # Step 2: Fetch news data
        news_data = fetch_news_data([asset], user_email)

        # Step 3: Analyse chart
        chart_analysis = analyse_image_from_file(chart_path, news_data)

        # Combine analysis
        return {
            "chart_analysis": chart_analysis,
            # "news_analysis": news_data
        }
    except Exception as e:
        print(f"Error in combined analysis: {e}")


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

class TraderDialogue:
    def __init__(self, asset: str, interval: str, num_days: int, max_messages: int = 6):
        self.asset = asset
        self.interval = interval
        self.num_days = num_days
        self.max_messages = max_messages
        self.messages: List[TraderMessage] = []

        # Initialize base data
        self.market_data = obtain_dataset(asset, interval, num_days)
        self.chart_path = generate_candlestick_chart(self.market_data)
        self.news_data = fetch_news_data([asset], user_email=None)

        # Initialize chart annotator
        self.chart_annotator = ChartAnnotator(self.market_data)

        # Define trader personalities
        self.trader_personalities = {
            "Trader1": {
                "style": "Conservative",
                "focus": "long-term trends and fundamental analysis",
                "risk_tolerance": "low"
            },
            "Trader2": {
                "style": "Aggressive",
                "focus": "short-term momentum and technical patterns",
                "risk_tolerance": "high"
            }
        }

    def _create_discussion_prompt(self, trader_id: str, previous_message: Optional[TraderMessage] = None) -> str:
        """Create a contextual prompt for regular discussion."""
        personality = self.trader_personalities[trader_id]

        if previous_message:
            base_prompt = f"""
            As a {personality['style']} trader focusing on {personality['focus']} with {personality['risk_tolerance']} risk tolerance,
            analyze this chart and respond to the following analysis from another trader:

            Previous Analysis: {previous_message.content}

            Consider:
            1. What points do you agree with and why?
            2. What factors might need additional consideration?
            3. How does your trading style inform your perspective?

            Aim to find common ground while highlighting important considerations.
            """
        else:
            base_prompt = f"""
            As a {personality['style']} trader focusing on {personality['focus']} with {personality['risk_tolerance']} risk tolerance,
            provide your initial analysis of this chart.
            """

        # Add market context
        current_close = self.market_data['Close'].iloc[-1].item()
        current_open = self.market_data['Open'].iloc[-1].item()
        price_change = current_close - current_open
        price_change_pct = (price_change / current_open) * 100

        market_context = f"""
        Current market context:
        - Price change: {price_change_pct:.2f}%
        - Current price: {current_close:.4f}
        """

        return base_prompt + "\n" + market_context

    def _create_consensus_prompt(self) -> str:
        """Create a prompt for the final consensus phase."""
        previous_analyses = "\n".join([
            f"{msg.trader_id}: {msg.content}"
            for msg in self.messages
        ])

        return f"""
        Review the following discussion about {self.asset}:

        {previous_analyses}

        As a group of traders, we need to reach a final consensus.
        Please provide specific levels in your analysis:
        1. Key support and resistance levels
        2. Suggested entry price
        3. Stop-loss level
        4. Target price
        5. Overall trend direction (uptrend/downtrend)

        Also include:
        1. Points of agreement between traders
        2. How different risk tolerances and trading styles are balanced
        3. Final recommendation that considers both perspectives
        4. Risk management suggestions

        Format your response as JSON with 'analysis' and 'recommendation' keys.
        Include numerical price levels in your analysis for chart annotation.
        """

    def _generate_response(self, trader_id: str, previous_message: Optional[TraderMessage] = None,
                         message_type: str = "discussion") -> str:
        """Generate a response based on message type."""
        if message_type == "consensus":
            prompt = self._create_consensus_prompt()
        else:
            prompt = self._create_discussion_prompt(trader_id, previous_message)

        modified_news = self.news_data.copy()
        modified_news['discussion_prompt'] = prompt

        return analyse_image_from_file(self.chart_path, modified_news)

    def conduct_dialogue(self) -> Tuple[List[TraderMessage], str]:
        """Conduct a dialogue and create annotated chart."""
        # Regular discussion phase
        discussion_messages = self.max_messages - 1

        # First message
        initial_message = TraderMessage(
            trader_id="Trader1",
            content=self._generate_response("Trader1"),
            message_type="discussion"
        )
        self.messages.append(initial_message)

        # Continue discussion
        current_msg_count = 1
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

        # Final consensus phase
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
        annotated_chart_path = self.chart_annotator.draw_annotated_chart(
            consensus.content,
            save_path=f"annotated_{self.asset}_chart.png"
        )

        return self.messages, annotated_chart_path

def run_trader_dialogue(asset: str, interval: str = '1h', num_days: int = 7, max_messages: int = 6):
    dialogue = TraderDialogue(asset, interval, num_days, max_messages)
    conversation, annotated_chart = dialogue.conduct_dialogue()

    print(f"\n=== Trading Discussion for {asset.upper()} ===\n")

    for i, msg in enumerate(conversation, 1):
        print(f"Message {i} - {msg.trader_id}:")
        if msg.message_type == "consensus":
            print("=== FINAL CONSENSUS ===")
        print(f"{msg.content}")
        if msg.responding_to:
            print(f"(Responding to {msg.responding_to})")
        print("\n" + "="*50 + "\n")

    print(f"Annotated chart saved as: {annotated_chart}")
    return conversation, annotated_chart


# @csrf_exempt
# def get_trader_analysis(request):
#     try:
#         if request.method == 'POST':
#             # Assuming raw JSON body is sent (e.g., Content-Type: application/json)
#             data = json.loads(request.body)
            
#             asset = data.get('asset', 'EURUSD')
#             interval = data.get('interval', '1h')
#             num_days = int(data.get('num_days', 7))
            
#             # asset = request.POST.get('asset', 'EURUSD')
#             # interval = request.POST.get('interval', '1h')
#             # num_days = int(request.POST.get('num_days', 7))
            
#             # Run the trader dialogue analysis
#             conversation, chart_path = run_trader_dialogue(asset, interval, num_days)
            
#             # Convert the conversation to a serializable format
#             conversation_data = []
#             for msg in conversation:
#                 if isinstance(msg.content, str):
#                     content = msg.content.replace('```json\n', '').replace('\n```', '')
#                     try:
#                         parsed_content = json.loads(content)
#                         if 'analysis' in parsed_content:
#                             if isinstance(parsed_content['analysis'], str):
#                                 parsed_content['analysis'] = parsed_content['analysis'][:1000]
#                             else:
#                                 parsed_content['analysis'] = str(parsed_content['analysis'])[:1000]
#                         content = parsed_content
#                     except json.JSONDecodeError:
#                         content = content[:1000]
#                 else:
#                     content = msg.content

#                 conversation_data.append({
#                     'trader_id': msg.trader_id,
#                     'content': content,
#                     'message_type': msg.message_type,
#                     'responding_to': msg.responding_to
#                 })

#             # Compress and resize the image
#             image = Image.open(chart_path)

#             # Convert to RGB if the image is in RGBA mode (to remove alpha channel)
#             if image.mode == 'RGBA':
#                 image = image.convert('RGB')

#             # Resize the image (e.g., reduce dimensions to 800x800)
#             max_size = (800, 800)
#             image.thumbnail(max_size)  # Resize to fit within 800x800

#             # Save the image to a BytesIO object with lower quality (50 or 60)
#             compressed_image_io = io.BytesIO()
#             image.save(compressed_image_io, format='JPEG', quality=50)  # Adjust quality as needed (lower = smaller file)
#             compressed_image_io.seek(0)

#             # Encode the compressed image to base64
#             encoded_image = base64.b64encode(compressed_image_io.read()).decode('utf-8')
            
#             # Clean up the image file
#             if os.path.exists(chart_path):
#                 os.remove(chart_path)
            
#             response_data = {
#                 'status': 'success',
#                 'conversation': conversation_data,
#                 'chart_image': encoded_image
#             }
            
#             return JsonResponse(response_data)
#         else:
#             return JsonResponse({
#                 'status': 'error',
#                 'message': 'Invalid request method.',
#                 'type': 'InvalidRequestMethod'
#             }, status=400)
            
#     except Exception as e:
#         return JsonResponse({
#             'status': 'error',
#             'message': str(e),
#             'type': type(e).__name__
#         })


from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from PIL import Image
import io
import base64
import os
from typing import Optional, Tuple, List
from dataclasses import dataclass


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

        # Initialize news data for both assets
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
                "bias": trader1_settings.get('bias', 'neutral'),
                "settings": trader1_settings,
                "data": self.trader1_data,
                "chart": self.trader1_chart
            },
            "Trader2": {
                "style": trader2_settings.get('style', 'Aggressive'),
                "focus": trader2_settings.get('focus', 'short-term momentum and technical patterns'),
                "risk_tolerance": trader2_settings.get('risk_tolerance', 'high'),
                "bias": trader2_settings.get('bias', 'neutral'),
                "settings": trader2_settings,
                "data": self.trader2_data,
                "chart": self.trader2_chart
            }
        }

    def _create_discussion_prompt(self, trader_id: str, previous_message: Optional[TraderMessage] = None) -> str:
        personality = self.trader_personalities[trader_id]
        settings = personality['settings']
        data = personality['data']

        base_prompt = f"""
        As a {personality['style']} trader with {personality['risk_tolerance']} risk tolerance,
        analyzing {settings['asset']} on the {settings['interval']} timeframe 
        with a {settings['numDays']}-day lookback period, focusing on {personality['focus']},
        and having a {personality['bias']} market bias,
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

            Aim to find common ground while highlighting important considerations from your perspective.
            """
        else:
            base_prompt += "provide your initial analysis of this chart."

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

        return f"""
        Review the following discussion about market analysis from different timeframes and trading styles:

        {previous_analyses}

        As a group of traders analyzing multiple timeframes, we need to reach a final consensus.
        Please provide specific levels in your analysis:
        1. Key support and resistance levels from both timeframes
        2. Suggested entry price considering both analyses
        3. Stop-loss level that respects both timeframes
        4. Target price based on both perspectives
        5. Overall trend direction (uptrend/downtrend) on each timeframe

        Also include:
        1. Points of agreement between different timeframe analyses
        2. How different risk tolerances and trading styles are balanced
        3. Final recommendation that considers both timeframes
        4. Risk management suggestions that account for multiple timeframe analysis

        Format your response as JSON with 'analysis' and 'recommendation' keys.
        Include numerical price levels in your analysis for chart annotation.
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
            
            # Initialize multi-trader dialogue
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
                                parsed_content['analysis'] = parsed_content['analysis'][:1000]
                            else:
                                parsed_content['analysis'] = str(parsed_content['analysis'])[:1000]
                        content = parsed_content
                    except json.JSONDecodeError:
                        content = content[:1000]
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
                        'bias': dialogue.trader_personalities[msg.trader_id]['bias'] if msg.trader_id in dialogue.trader_personalities else None
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
                }
            }
            
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
