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
    path('fetch_news_data/<str:user_email>', views.fetch_news_data, name='fetch_news_data'),
    # path('save_news_data', views.save_news_data, name='save_news_data'),
    path('create-bot/<str:type_1>/<str:type_2>/<int:ma1>/<int:ma2>/<str:dataframe>/<str:backtest_period>', views.moving_average_bot, name='create-bot'),
    path('create-bot/bbands/<int:length>/<int:std>/<str:dataframe>/<str:backtest_period>', views.bbands_bot, name='create-bot-bbands'),
    path('create-bot/rsi/<str:length>/<str:overbought_level>/<str:oversold_level>/<str:dataframe>/<str:backtest_period>', views.rsi_bot, name='create-bot-rsi'),
    path('create-bot/momentum/<str:dataframe>/<str:backtest_period>', views.momentum_bot, name='create-bot-momentum'),
    path('create-bot/candlesticks/<str:dataframe>/<str:backtest_period>', views.candlesticks_bot, name='create-bot-candlesticks'),
    path('api-call/<str:asset>', views.api_call, name='api-call'),
    path('api-test', views.api_test, name='api-test'),
    path('new-test', views.new_test, name='new-test'),
    path('download-mq4/<str:bot>', views.download_mq4_file, name='download_mq4_file'),
    path('process-image', views.process_image, name='process_image'),
    path('chosen-models/<str:user_email>/<int:magic_number>', views.chosen_models, name='chosen_models'),
    path('run-bot/<str:user_email>/<int:magic_number>/<str:asset>', views.run_bot, name='run-bot'),
    path('run-backtest/<str:dataframe>/<str:backtest_period>', views.run_backtest, name='run-backtest'),
    path('update-news-data/<str:user_email>', views.update_news_data, name='update-news-data'),
    path('interest-rates-data', views.interest_rates, name='interest-rates-data'),
    path('genesys', views.genesys, name='genesys'),
    path('save-dataset/<str:dataset>', views.save_dataset, name='save-dataset'),
    path('split-dataset', views.split_dataset, name='split-dataset'),
    path('set-init-capital', views.set_init_capital, name='set-init-capital'),
    path('genesys-live/<str:identifier>/<float:equity>', views.genesys_live, name='genesys-live'),

    path('contact-us', views.contact_us, name='contact-us'),
    path('book-order', views.book_order, name='book-order'),
    path('api/register/', UserRegistrationView.as_view(), name='user-register'),
    path('api/login/', views.user_login, name='user-login'),
    path('api/csrf_token/', views.get_csrf_token, name='get_csrf_token'),
    # path('fetch_user_email', views.fetch_user_email, name='fetch_user_email')
]


