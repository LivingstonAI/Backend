from django.urls import path
from .views import *
from . import views
 
urlpatterns = [
    # ... other URL patterns
    path('register/', UserRegistrationView.as_view(), name='user-registration'),
    path('check_email/', views.check_email, name='check_email'),
    path('tell_us_more/create/', views.TellUsMoreCreateView.as_view(), name='tell_us_more_create'),
    path('login/', UserLoginView.as_view(), name='user-login'),
    path('new_trade/', TradeView.as_view(), name='new_trade'),
    path('all_trades/<str:email>/', views.all_trades, name='all_trades'),
    path('full_trade/<int:trade_id>/', views.full_trade, name='full_trade'),
    path('user_overview/<str:user_email>/', views.user_overview, name='overview'),
    path('save_journal/<str:user_email>/', views.save_journal, name='save_journal'),
    path('all_journals/<str:user_email>/', views.fetch_journals, name='all_journals'),
    path('view_journal/<int:journal_id>/', views.view_journal, name='view_journal'),
    path('upcoming_news/<str:user_email>/', views.upcoming_news, name='upcoming_news'),
    path('get_user_data/<str:user_email>/', views.get_user_data, name='user_data'),
    path('save_conversation/<str:user_email>/<str:identifier>/', views.save_conversation, name='save_conversation'),
    path('fetch_conversations/<str:user_email>/', views.fetch_conversations, name='fetch_conversations'),
    path('fetch_conversation/<str:conversation_id>/', views.fetch_conversation, name='fetch_conversation'),
    path('update_conversation/<str:conversation_id>/', views.update_conversation, name='update_conversation'),
    path('delete_conversation/<str:conversation_id>/', views.delete_conversation, name='delete_conversation'),
    path('get_openai_key', views.get_openai_key, name='get_openai_key'),
    path('update_user_data/<str:user_email>/', views.update_tell_us_more, name='update_user_data'),
    path('update_assets/<str:user_email>/', views.update_user_assets, name='update_assets'),
    path('fetch_news_data/', views.fetch_news_data, name='fetch_news_data'),
    # path('save_news_data', views.save_news_data, name='save_news_data'),
    path('create-bot/<str:type_1>/<str:type_2>/<int:ma1>/<int:ma2>', views.moving_average_bot, name='create-bot'),
    path('create-bot/bbands/<int:length>/<int:std>', views.bbands_bot, name='create-bot-bbands'),
    path('create-bot/rsi/<int:length>/<int:overbought_level>/<int:oversold_level>/', views.rsi_bot, name='create-bot-rsi'),
    path('create-bot/momentum', views.momentum_bot, name='create-bot-momentum')
    # path('fetch_user_email', views.fetch_user_email, name='fetch_user_email')
]


