from django.urls import path
from .views import *
from . import views
 
urlpatterns = [
    # ... other URL patterns
#     path('register/', UserRegistrationView.as_view(), name='user-registration'),
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
    path('fetch_user_news_data/<str:user_email>', views.fetch_user_news_data, name='fetch_news_data'),
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
    path('genesys-live/<int:identifier>/<int:num_positions>/<str:asset>/<str:interval>/<str:order_ticket>/<str:bot_id>', views.genesys_live, name='genesys-live'),
    path('save-genesys-model', views.save_genesys_model, name='save-genesys-model'),
    path('test-cnn/<str:asset>/<str:interval>/<int:num_days>', views.test_cnn, name='test-cnn'),
    path('save-new-trade-model/<int:model_id>/<int:initial_equity>/<str:order_ticket>/<str:asset>/<str:volume>/<str:type_of_trade>/<str:timeframe>/<str:bot_id>', views.save_new_trade_model, name='save-new-trade-model'),
    path('update-trade-model/<int:model_id>/<str:order_ticket>/<str:profit>', views.update_trade_model, name='update-trade-model'),
    path('get-model-performance', views.get_model_performance, name='get-model-performance'),
    path('delete-bot/<str:bot_id>', views.delete_unique_bot, name='delete-bot'),
    path('clear-bots', views.clear_stuff, name='clear-bots'),
    path('test-date/<str:asset>', views.test_date, name='test-date'),
    path('daily-brief', views.daily_brief, name='daily-brief'),
    path('fetch-daily-brief-data', views.fetch_daily_brief_data, name='fetch-daily-brief-data'),
    path('get-user-assets', views.get_user_assets, name='get-user-assets'),
    path('fetch-asset-data/<str:asset>', views.fetch_asset_data, name='fetch-asset-data'),
    path('fetch-asset-data-from-models/<str:asset>', views.fetch_asset_data_from_models, name='fetch-asset-data-from-models'),
    path('get-asset-summary/<str:asset>', views.get_asset_summary, name='get-asset-summary'),
    path('reflections-summary/<str:asset>', views.reflections_summary, name='reflections-summary'),
    path('generate-cot-data', views.generate_cot_data, name='generate-cot-data'),
    path('create-chill-data', views.create_chill_data, name='create-chill-data'),
    path('fetch-chill-sections', views.fetch_chill_sections, name='fetch-chill-sections'),
    path('fetch-chill-data', views.fetch_chill_data, name='fetch-chill-data'), 
    path('edit-chill-data', views.edit_chill_data, name='edit-chill-data'),
    path('delete-chill-entry', views.delete_chill_entry, name='delete-chill-entry'),
    path('fetch-trading-images', views.fetch_trading_images, name='fetch-trading-images'),
    path('alert-bot', views.alert_bot, name='alert-bot'),
    path('create-finetuning-data', views.create_finetuning_data, name='create-finetuning-data'),
    path('create-combined-finetuning-data', views.create_combined_finetuning_data, name='create-combined-finetuning-data'),
    path('create-image-finetuning-data', views.create_image_finetuning_data, name='create-image-finetuning-data'),
    path('accounts/', views.get_accounts, name='get_accounts'),
    path('create_account/', views.create_account, name='create_account'),
    path('delete_account/<int:account_id>/', views.delete_account, name='delete_account'),
    path('accounts/update/', views.update_account, name='update_account'),
    path('get-trading-analytics', views.get_trading_analytics, name='get-trading-analytics'),
    path('create-new-trade-data', views.create_new_trade_data, name='create-new-trade-data'),
    path('fetch-trading-data', views.fetch_trading_data, name='fetch-trading-data'),
    path('fetch-account-data/', views.fetch_account_data, name='fetch-account-data'),
    path('set-daily-brief-assets', views.set_daily_brief_assets, name='set-daily-brief-assets'),
    path('time-trading-analytics', views.time_trading_analytics, name='trading_analytics'),
    # path('api/trader-analysis/', views.trader_analysis, name='trader-analysis'),
    path('api/trader-analysis/', views.get_trader_analysis, name='trader-analysis'),
    path('save-backtest-model-data', views.save_backtest_model_data, name='save-backtest-model-data'),
    path('trigger-backtest', views.trigger_backtest, name='trigger-backtest'),
    path('fetch-backtested-results', views.fetch_backtested_results, name='fetch-backtested-results'),
    path('delete-backtest-model', views.delete_backtest_model, name='delete-backtest-model'),
    path('generate-idea', views.generate_idea, name='generate-idea'),
    path('fetch-ideas', views.fetch_ideas, name='fetch-ideas'),
    path('delete-idea', views.delete_idea, name='delete-idea'),
    path('update-idea-tracker', views.update_idea_tracker, name='update-idea-tracker'),
    path('update-idea', views.update_idea, name='update-idea'),
    path('ai-account-summary', views.get_ai_account_summary, name='ai-account-summary'),
    path('save-quiz', views.save_quiz, name='save-quiz'),
    path('fetch-saved-quizzes', views.fetch_saved_quizzes, name='fetch-saved-quizzes'),
    path('delete-quiz/<int:quiz_id>', views.delete_quiz, name='delete-quiz'),
    path('save-music', views.save_music, name='save-music'),
    path('fetch-music', views.fetch_music, name='fetch-music'),
    path('stream-music/<int:song_id>', views.stream_music, name='stream_music'),
    path('delete-music', views.delete_music, name='delete-music'),
    path('fetch-asset-update/', views.fetch_asset_update, name='fetch-asset-update'),
    path('get-tracked-assets/', views.get_tracked_assets, name='get-tracked-assets'),
    path('add-tracked-asset/', views.add_tracked_asset, name='add-tracked-asset'),
    path('remove-tracked-asset/', views.remove_tracked_asset, name='remove-tracked-asset'),
    path('api/trade-ideas/', views.get_trade_ideas, name='get_trade_ideas'),
    path('api/trade-ideas/create/', views.create_trade_idea, name='create_trade_idea'),
    path('api/trade-ideas/update/<int:id>/', views.update_trade_idea, name='update_trade_idea'),
    path('api/trade-ideas/delete/<int:id>/', views.delete_trade_idea, name='delete_trade_idea'),
    path('api/prop-firms/', views.prop_firm_list, name='prop_firm_list'),
    path('api/prop-firms/<int:firm_id>/', views.prop_firm_detail, name='prop_firm_detail'),
    path('api/prop-metrics/', views.metrics_list, name='metrics_list'),
    path('api/prop-metrics/<int:metric_id>/', views.metric_detail, name='metric_detail'),
    path('api/economic-events/', views.economic_events_list, name='economic-events-list'),
    path('api/economic-events/<int:pk>/', views.economic_event_detail, name='economic-event-detail'),
    path('api/data-calendar-economic-events/', views.data_calendar_economic_events_list, name='economic-events-list'),
    path('api/data-calendar-economic-events/<int:event_id>/', views.data_calendar_economic_event_detail, name='economic-event-detail'),
    path('api/unique-economic-events/', views.unique_economic_events_list, name='unique_economic_events_list'),
    path('api/event-history/<str:event_name>/', views.event_history, name='event-history'),
    path('api/cot/', views.generate_econ_cot_data, name='generate_econ_cot_data'),
    path('api/generate_econ_ai_summary/', views.generate_econ_ai_summary, name='generate_econ_ai_summary'),
    # path('save_forex_factory_news', views.save_forex_factory_news, name='save_forex_factory_news'),
    # path('get_forex_factory_events', views.get_forex_factory_events, name='get_forex_factory_events'),
    path('save_forex_factory_news', views.save_forex_factory_news, name='save_forex_factory_news'),
    path('fetch_news_data_api', views.fetch_news_data_api, name='fetch_news_data_api'),
    # Get all trades for a specific account
    path('api/trades-calendar/', views.get_trades_by_account_calendar, name='get_trades_by_account'),
    # Get trades within a date range
    path('api/trades/date-range/', views.get_trades_by_date_range_calendar, name='get_trades_by_date_range'),
    # Get trade summary statistics
    path('api/trades/summary-calendar/', views.get_trade_summary_calendar, name='get_trade_summary'),
    # Create a new trade
    path('api/trades/create-calendar/', views.create_trade_calendar, name='create_trade'),
    path('paper-gpt/', views.paper_gpt, name='paper_gpt'),
    path('paper-gpt/categories/', views.get_categories, name='get_categories'),
    path('paper-gpt/<uuid:paper_id>/', views.paper_detail, name='paper_detail'),
    path('extract-pdf-text/', views.extract_pdf_text, name='extract_pdf_text'),
    path('generate-summary/', views.generate_paper_summary, name='generate_summary'),
    path('api/v1/economics/fetch-currency-economic-insights/<str:currency_code>/', 
            views.retrieve_economic_data_for_selected_currency, 
            name='retrieve_economic_data_for_selected_currency'),
    path('api/v1/economics/generate-dynamic-chart/', views.generate_dynamic_chart, name='generate_dynamic_chart'),
    # TraderGPT Advanced Forex Analysis URLs
    path('api/trader-gpt-advanced-forex-analysis/', views.advanced_trader_gpt_forex_analysis_endpoint, name='advanced_trader_gpt_forex_analysis'),
    path('api/trader-gpt-analysis-history/', views.fetch_trader_gpt_analysis_history_endpoint, name='trader_gpt_analysis_history'),
    # TraderGPT Analysis URLs
    path('api/fetch-watched-trading-assets/', views.fetch_watched_trading_assets_view, name='fetch_watched_trading_assets'),
    path('api/add-trading-asset-to-watch/', views.add_trading_asset_to_watch_view, name='add_trading_asset_to_watch'),
    path('api/remove-watched-trading-asset/', views.remove_watched_trading_asset_view, name='remove_watched_trading_asset'),
    path('api/fetch-trader-gpt-analyses/', views.fetch_trader_gpt_analyses_view, name='fetch_trader_gpt_analyses'),
    path('api/run-fresh-trader-analysis/', views.run_fresh_trader_analysis_view, name='run_fresh_trader_analysis'),
    path('api/trigger-bulk-analysis/', views.trigger_bulk_analysis_view, name='trigger_bulk_analysis'),
    path('api/manage-scheduler/', views.manage_scheduler_view, name='manage_scheduler'),
    # Scheduler Control URLs
    path('api/start-analysis-scheduler/', views.start_scheduler_view, name='start_analysis_scheduler'),
    path('api/stop-analysis-scheduler/', views.stop_scheduler_view, name='stop_analysis_scheduler'),
    path('api/scheduler-status/', views.scheduler_status_view, name='scheduler_status'),
    path('api/ai-council/trigger-conversation/', 
         views.run_manual_council_conversation_view, 
         name='trigger_ai_council_conversation'),
    
    # Get all council conversations (with pagination)
    path('api/ai-council/conversations/', 
         views.get_council_conversations_view, 
         name='get_ai_council_conversations'),
    
    # Get specific conversation details
    path('api/ai-council/conversations/<str:conversation_id>/', 
         views.get_council_conversation_details_view, 
         name='get_ai_council_conversation_details'),
     
     # Firm compliance endpoints
    path('api/firm-compliance/', views.firm_compliance_list, name='firm_compliance_list'),
    path('api/firm-compliance/<uuid:compliance_id>/', views.firm_compliance_detail, name='firm_compliance_detail'),

    # Economic Strength Index  
    path('api/economic-strength-index/', views.economic_strength_index, name='economic-strength-index'),

    # Main API endpoints for ML model entries
    path('snowai-research-logbook/api/ml-entries/', 
         views.snowai_research_logbook_api_entries, 
         name='snowai_research_logbook_api_entries'),
    
    path('snowai-research-logbook/api/ml-entries/<int:entry_id>/', 
         views.snowai_research_logbook_api_entry_detail, 
         name='snowai_research_logbook_api_entry_detail'),
    
    # Analytics and statistics endpoint
    path('snowai-research-logbook/api/analytics/', 
         views.snowai_research_logbook_api_analytics, 
         name='snowai_research_logbook_api_analytics'),
    
    # Tags endpoint for autocomplete and filtering
    path('snowai-research-logbook/api/tags/', 
         views.snowai_research_logbook_api_tags, 
         name='snowai_research_logbook_api_tags'),

     path('check_fingerprint_status/', views.check_fingerprint_status, name='check_fingerprint_status'),
     path('register_fingerprint/', views.register_fingerprint_backend, name='register_fingerprint_backend'),
     path('reset_fingerprint/', views.reset_fingerprint_backend, name='reset_fingerprint_backend'),


     # path('trader_history_gpt_summary/', snowai_trader_history_gpt_summary_endpoint, name='snowai_trader_history_gpt_summary'),
     # # path('trader_history_gpt_chat/', snowai_trader_history_gpt_chat_endpoint, name='snowai_trader_history_gpt_chat'),
     # path('macro_gpt_summary/', snowai_macro_gpt_summary_endpoint, name='snowai_macro_gpt_summary'),
     # # path('macro_gpt_chat/', snowai_macro_gpt_chat_endpoint, name='snowai_macro_gpt_chat'),
     # path('idea_gpt_summary/', snowai_idea_gpt_summary_endpoint, name='snowai_idea_gpt_summary'),
     # # path('idea_gpt_chat/', snowai_idea_gpt_chat_endpoint, name='snowai_idea_gpt_chat'),
     # path('backtesting_gpt_summary/', snowai_backtesting_gpt_summary_endpoint, name='snowai_backtesting_gpt_summary'),
     # # path('backtesting_gpt_chat/', snowai_backtesting_gpt_chat_endpoint, name='snowai_backtesting_gpt_chat'),
     # path('paper_gpt_summary/', snowai_paper_gpt_summary_endpoint, name='snowai_paper_gpt_summary'),
     # # path('paper_gpt_chat/', snowai_paper_gpt_chat_endpoint, name='snowai_paper_gpt_chat'),
     # path('research_gpt_summary/', snowai_research_gpt_summary_endpoint, name='snowai_research_gpt_summary'),
     # # path('research_gpt_chat/', snowai_research_gpt_chat_endpoint, name='snowai_research_gpt_chat'),

     path('get_existing_summary/<str:gpt_type>/', views.get_existing_summary, name='get_existing_summary'),
    
     # Optional: Manual trigger endpoint for testing
     path('trigger_summaries/', views.manual_trigger_summaries, name='manual_trigger_summaries'),


     # Conversation memory endpoints
    path('get_conversation_history/<str:gpt_system>/', views.get_conversation_history, name='get_conversation_history'),
    path('clear_conversation_history/<str:gpt_system>/', views.clear_conversation_history, name='clear_conversation_history'),
    
    # Updated chat endpoints (replace your existing ones)
    path('paper_gpt_chat/', views.snowai_paper_gpt_chat_endpoint, name='paper_gpt_chat'),
    path('backtesting_gpt_chat/', views.snowai_backtesting_gpt_chat_endpoint, name='backtesting_gpt_chat'),
    path('research_gpt_chat/', views.snowai_research_gpt_chat_endpoint, name='research_gpt_chat'),
    path('trader_history_gpt_chat/', views.snowai_trader_history_gpt_chat_endpoint, name='trader_history_gpt_chat'),
    path('idea_gpt_chat/', views.snowai_idea_gpt_chat_endpoint, name='idea_gpt_chat'),
    path('macro_gpt_chat/', views.snowai_macro_gpt_chat_endpoint, name='macro_gpt_chat'),

    # Custom interest rates endpoints with unique names to avoid conflicts
    path('api/v2024/custom-global-interest-rates-database/', 
         views.fetch_custom_global_interest_rates_data_v2024, 
         name='custom_global_interest_rates_v2024'),
    
    path('api/v2024/custom-global-interest-rates-extended-cny/', 
         views.fetch_custom_global_interest_rates_extended_cny_v2024, 
         name='custom_global_interest_rates_extended_cny_v2024'),

     path('api/economic-data-map/', views.get_country_economic_data, name='get_country_economic_data'),

     # GPT Discussion URLs
     path('trigger_manual_gpt_discussion/', views.trigger_manual_gpt_discussion, name='trigger_manual_gpt_discussion'),
     path('get_current_gpt_discussion/', views.get_current_gpt_discussion, name='get_current_gpt_discussion'),

     path('api/snowai-accounts-deep-analysis/', views.fetch_snowai_accounts_for_deep_analysis, name='fetch_snowai_accounts_for_deep_analysis'),
     path('api/deep-account-diagnostics/<int:account_id>/', views.calculate_deep_account_performance_metrics, name='calculate_deep_account_performance_metrics'),
     path('api/ai-enhanced-diagnostics/<int:account_id>/', views.generate_ai_enhanced_account_diagnostics, name='generate_ai_enhanced_account_diagnostics'),



     path('snowai_extract_youtube_transcript_from_url', 
         views.snowai_extract_youtube_transcript_from_url, 
         name='snowai_extract_youtube_transcript_from_url'),
    
    # Get all saved transcripts with pagination and filtering
    path('snowai_get_all_saved_transcripts', 
         views.snowai_get_all_saved_transcripts, 
         name='snowai_get_all_saved_transcripts'),
    
    # Get single transcript details
    path('snowai_get_single_transcript_details/<str:transcript_id>', 
         views.snowai_get_single_transcript_details, 
         name='snowai_get_single_transcript_details'),
    
    # Delete specific transcript record
    path('snowai_delete_transcript_record/<str:transcript_id>', 
         views.snowai_delete_transcript_record, 
         name='snowai_delete_transcript_record'),
    
    # Update transcript metadata
    path('snowai_update_transcript_metadata/<str:transcript_id>', 
         views.snowai_update_transcript_metadata, 
         name='snowai_update_transcript_metadata'),

     
     # AI Analysis endpoints
    path('snowai-transcript-analysis-dashboard-data/', 
         views.snowai_transcript_analysis_dashboard_data_endpoint, 
         name='snowai_transcript_analysis_dashboard_data'),
         
    path('snowai-trigger-transcript-analysis/', 
         views.snowai_trigger_transcript_analysis_endpoint, 
         name='snowai_trigger_transcript_analysis'),
         
    path('snowai-transcript-analysis-details/<str:transcript_uuid>/', 
         views.snowai_get_transcript_analysis_details_endpoint, 
         name='snowai_transcript_analysis_details'),
         
    path('snowai-delete-transcript-analysis/<str:analysis_uuid>/', 
         views.snowai_delete_transcript_analysis_endpoint, 
         name='snowai_delete_transcript_analysis'),

     path('api/snowai-market-ohlc/', views.snowai_fetch_market_ohlc_data, name='snowai_market_ohlc'),

     path('api/livingston-ai-fetch-latest-council-discussion/', 
         views.fetch_latest_ai_council_discussion_for_livingston, 
         name='livingston_fetch_council'),

     path('api/snowai-time-separators/', views.snowai_fetch_time_separators, name='snowai_time_separators'),

     path('api/snowai-asset-correlation-data/', views.snowai_asset_correlation_get_data, name='snowai_asset_correlation_get_data'),
     path('api/snowai-asset-correlation-correlations/', views.snowai_asset_correlation_calculate_correlations, name='snowai_asset_correlation_calculate_correlations'),
     path('api/snowai-asset-correlation-classes/', views.snowai_asset_correlation_get_all_classes, name='snowai_asset_correlation_get_all_classes'),

     path('api/mss/calculate/', views.calculate_market_stability_score, name='calculate_mss'),
     path('api/mss/historical/', views.get_mss_historical_data, name='mss_historical'),
     path('api/mss/asset-lists/', views.get_predefined_asset_lists, name='mss_asset_lists'),

     path('snowai/hedge-funds/', views.snowai_get_all_hedge_funds, name='snowai_get_all_hedge_funds'),
     path('snowai/hedge-funds/create/', views.snowai_create_hedge_fund, name='snowai_create_hedge_fund'),
     path('snowai/hedge-funds/<int:fund_id>/update/', views.snowai_update_hedge_fund, name='snowai_update_hedge_fund'),
     path('snowai/hedge-funds/<int:fund_id>/delete/', views.snowai_delete_hedge_fund, name='snowai_delete_hedge_fund'),
    
     # Key People
     path('snowai/hedge-funds/<int:fund_id>/key-person/add/', views.snowai_add_key_person, name='snowai_add_key_person'),
     path('snowai/hedge-funds/key-person/<int:person_id>/delete/', views.snowai_delete_key_person, name='snowai_delete_key_person'),
    
     # Resources
     path('snowai/hedge-funds/<int:fund_id>/resource/add/', views.snowai_add_resource, name='snowai_add_resource'),
     path('snowai/hedge-funds/resource/<int:resource_id>/delete/', views.snowai_delete_resource, name='snowai_delete_resource'),
    
     # Performance
     path('snowai/hedge-funds/<int:fund_id>/performance/add/', views.snowai_add_performance, name='snowai_add_performance'),
     path('snowai/hedge-funds/performance/<int:performance_id>/delete/', views.snowai_delete_performance, name='snowai_delete_performance'),



     # Add these to your urlpatterns
     path('snowai/hedge-funds/<int:fund_id>/update/', snowai_update_hedge_fund),
     path('snowai/hedge-funds/key-person/<int:person_id>/update/', snowai_update_key_person),
     path('snowai/hedge-funds/resource/<int:resource_id>/update/', snowai_update_resource),
     path('snowai/hedge-funds/performance/<int:performance_id>/update/', snowai_update_performance),

     path('api/snowai_intermarket_council_driven_asset_sentiment_analysis_v2/', views.snowai_intermarket_council_driven_asset_sentiment_analysis_v2),
     path('api/snowai_advanced_volume_proportion_analyzer_for_trading_assets_v2/', views.snowai_advanced_volume_proportion_analyzer_for_trading_assets_v2),
     path('api/snowai_fetch_latest_council_discussion_summary_for_frontend_v2/', views.snowai_fetch_latest_council_discussion_summary_for_frontend_v2),

     path('api/snowai_save_asset_bias_recommendation_v2/', 
         snowai_save_asset_bias_recommendation_v2, 
         name='save_asset_bias'),
    
     path('api/snowai_get_all_saved_asset_biases_v2/', 
         snowai_get_all_saved_asset_biases_v2, 
         name='get_all_biases'),
    
     path('api/snowai_get_asset_bias_by_name_v2/', 
         snowai_get_asset_bias_by_name_v2, 
         name='get_asset_bias'),
    
     path('api/snowai_delete_asset_bias_v2/', 
         snowai_delete_asset_bias_v2, 
         name='delete_asset_bias'),

     path('calculate_trade_probability', calculate_trade_probability, name='calculate_trade_probability'),

     path('simulate-trading-performance', simulate_trading_performance, name='simulate_trading_performance'),

     path(
        "obliterate-latest/<int:count>/",
        views.obliterate_latest_backtest_results,
        name="obliterate_latest_backtest_results"
    ),


    # Multi-Account Analytics endpoints
    path('fetch_multi_account_performance_overview_data/', 
         views.fetch_multi_account_performance_overview_data, 
         name='fetch_multi_account_performance_overview_data'),
    
    path('fetch_account_equity_curve_progression_data/<int:account_id>/', 
         views.fetch_account_equity_curve_progression_data, 
         name='fetch_account_equity_curve_progression_data'),
    
    path('fetch_all_accounts_equity_curves_comparison_data/', 
         views.fetch_all_accounts_equity_curves_comparison_data, 
         name='fetch_all_accounts_equity_curves_comparison_data'),
    
    path('execute_portfolio_monte_carlo_risk_simulation/', 
         views.execute_portfolio_monte_carlo_risk_simulation, 
         name='execute_portfolio_monte_carlo_risk_simulation'),

     path('api/snowai-video-categories/', views.fetch_snowai_video_categories, name='fetch_snowai_video_categories'),
     path('api/snowai-video-categories/create/', views.create_snowai_video_category, name='create_snowai_video_category'),
     path('api/snowai-video-entries/', views.fetch_snowai_video_entries, name='fetch_snowai_video_entries'),
     path('api/snowai-video-entries/create/', views.create_snowai_video_entry, name='create_snowai_video_entry'),
     path('api/snowai-video-entries/<int:video_id>/update/', views.update_snowai_video_entry, name='update_snowai_video_entry'),
     path('api/snowai-video-entries/<int:video_id>/delete/', views.delete_snowai_video_entry, name='delete_snowai_video_entry'),

     path('api/snowai_stock_screener_fetch_data/', views.snowai_stock_screener_fetch_data, name='snowai_stock_screener_fetch_data'),

     path('api/snowai-trading-weights/save/', views.snowai_save_trading_weights, name='snowai_save_weights'),
     path('api/snowai-trading-weights/load/<str:agent_name>/', views.snowai_load_trading_weights, name='snowai_load_weights'),
     path('api/snowai-trading-weights/list/', views.snowai_list_trading_weights, name='snowai_list_weights'),
     path('api/snowai-trading-weights/delete/<str:agent_name>/', views.snowai_delete_trading_weights, name='snowai_delete_weights'),
     path('api/snowai-debug-weights/', views.snowai_debug_weights, name='snowai_debug_weights'),

     # ===== SnowAI Forward Testing Model Endpoints =====
    
     # Save or update a forward testing model
     path(
          'snowai-forward-testing/save-model/', 
          views.save_snowai_forward_testing_model_endpoint, 
          name='save_snowai_forward_testing_model'
     ),
     
     # Retrieve a specific model by ID
     path(
          'snowai-forward-testing/get-model/<str:model_id>/', 
          views.retrieve_snowai_forward_testing_model_endpoint, 
          name='retrieve_snowai_forward_testing_model'
     ),
     
     # List all saved models
     path(
          'snowai-forward-testing/list-models/', 
          views.list_all_snowai_forward_testing_models_endpoint, 
          name='list_all_snowai_forward_testing_models'
     ),
     
     # Delete a specific model
     path(
          'snowai-forward-testing/delete-model/<str:model_id>/', 
          views.delete_snowai_forward_testing_model_endpoint, 
          name='delete_snowai_forward_testing_model'
     ),

     path('api/snowai-models/', views.snowai_models_list, name='snowai_models_list'),
     path('api/snowai-models/<int:model_id>/', views.snowai_model_detail, name='snowai_model_detail'),
     path('api/snowai-available-models/', views.snowai_available_models, name='snowai_available_models'),
     path('api/snowai-chart-data-with-positions/<int:model_id>/', views.snowai_fetch_chart_data_with_positions_endpoint, name='snowai_chart_data_with_positions'),
     path('api/snowai-trading-weights/debug-predictions-simple/', 
         views.snowai_debug_predictions_simple, 
         name='snowai_debug_predictions_simple'),

     path('api/snowai-trading-weights/debug-weight-shapes/', 
     views.snowai_debug_weight_shapes, 
     name='snowai_debug_weight_shapes'),

     path('api/detect-trend/', views.detect_trend_endpoint, name='detect_trend'),

     path('snow-ai/neuro-command/receive/', views.receive_sovereign_neuro_command_v1, name='neuro_command_receive'),

     path('api/snowai-sandbox/train/', views.snowai_sandbox_train, name='snowai_sandbox_train'),
     path('api/snowai-sandbox/status/<str:session_id>/', views.snowai_sandbox_status, name='snowai_sandbox_status'),
     path('api/snowai-sandbox/pause/<str:session_id>/', snowai_sandbox_pause),
     path('api/snowai-sandbox/resume/<str:session_id>/', snowai_sandbox_resume),
     path('api/snowai-sandbox/checkpoint/<str:session_id>/', snowai_sandbox_save_checkpoint),
     path('api/snowai-sandbox/checkpoints/', snowai_sandbox_list_checkpoints),

     path('api/mss-hyper-volumetric-relativistic-analyzer/', views.mss_hyper_volumetric_relativistic_analyzer, name='mss-hyper-volumetric-relativistic-analyzer'),
     path('api/mss-quantum-sector-momentum-flux-analyzer/', views.mss_quantum_sector_momentum_flux_analyzer, name='mss-quantum-sector-momentum-flux-analyzer'),
     path('api/mss-stock-sector-relativistic-performance-comparator/', views.mss_stock_sector_relativistic_performance_comparator, name='mss-stock-sector-relativistic-performance-comparator'),
     path('api/mss-stock-sector-identifier/', views.mss_stock_sector_identifier, name='mss-stock-sector-identifier'),
     path('api/monte-carlo-prediction/', views.mss_quantum_probabilistic_monte_carlo_forecaster_api, name='monte_carlo_prediction'),

     path('snowai_poi_create_person_unique_v1/', views.snowai_poi_create_person_unique_v1, name='snowai_poi_create_person_unique_v1'),
     path('snowai_poi_get_all_people_unique_v1/', views.snowai_poi_get_all_people_unique_v1, name='snowai_poi_get_all_people_unique_v1'),
     path('snowai_poi_get_person_unique_v1/<str:person_id>/', views.snowai_poi_get_person_unique_v1, name='snowai_poi_get_person_unique_v1'),
     path('snowai_poi_update_person_unique_v1/<str:person_id>/', views.snowai_poi_update_person_unique_v1, name='snowai_poi_update_person_unique_v1'),
     path('snowai_poi_delete_person_unique_v1/<str:person_id>/', views.snowai_poi_delete_person_unique_v1, name='snowai_poi_delete_person_unique_v1'),

     path('api/mss-quantum-retracement-fibonacci-entry-optimizer/', 
         views.mss_quantum_retracement_fibonacci_entry_optimizer, 
         name='mss_retracement_optimizer'),

     path('api/mss-sector-trend-elasticity-momentum-analyzer/', 
     views.mss_sector_trend_elasticity_momentum_analyzer, 
     name='sector_elasticity'),

     path('api/mss-trend-elasticity-analyzer/', views.mss_trend_elasticity_analyzer, name='mss_trend_elasticity_analyzer'),

     path('api/detect-early-trend-momentum/', views.detect_early_trend_momentum, name='detect_early_trend_momentum'),

     path('api/detect-trend-emergence/', views.detect_trend_emergence, name='detect_trend_emergence'),

     path('api/mss-analyze-trend-duration-timeline/', views.mss_analyze_trend_duration_timeline, name='mss_trend_duration'),

     path('api/mss-calculate-average-daily-range-projections/', views.mss_calculate_average_daily_range_projections, name='mss_adr_projections'),

     path('api/mss-estimate-price-target-timeline/', views.mss_estimate_price_target_timeline, name='mss_price_target_timeline'),

     # Run new backtest
     path('api/snowai-backtest/run/', views.snowai_backtest_run, name='snowai_backtest_run'),
     
     # Check backtest status
     path('api/snowai-backtest/status/<uuid:session_id>/', views.snowai_backtest_status, name='snowai_backtest_status'),
     
     # List all results
     path('api/snowai-backtest/results/', views.snowai_backtest_results_list, name='snowai_backtest_results_list'),
     
     # Get specific result detail
     path('api/snowai-backtest/result/<uuid:result_id>/', views.snowai_backtest_result_detail, name='snowai_backtest_result_detail'),

     path('api/mss-fetch-chart-data-for-visualization/', views.mss_fetch_chart_data_for_visualization, name='mss_chart_data'),
      # Mean Reversion & Regime Detection
     path('api/mss-mean-reversion-regime-detector-v2/', views.mss_mean_reversion_regime_detector_v2, name='mss_mean_reversion_regime_detector_v2'),
     
     # Sector Peers Normalized Index
     path('api/mss-sector-peers-normalized-index-v2/', views.mss_sector_peers_normalized_index_v2, name='mss_sector_peers_normalized_index_v2'),

     path('api/mss-generate-chart-context-for-ai-v2/', views.mss_generate_chart_context_for_ai_v2, name='mss_generate_chart_context_for_ai_v2'),

     # ============================================================
     # NEW URL ROUTES — Commodity / Sector Correlation Analysis
     # ============================================================

     # Bulk: Commodities vs Materials sector stocks
     path('api/mss-commodity-vs-materials-analyzer/', views.mss_commodity_vs_materials_analyzer, name='mss-commodity-vs-materials-analyzer'),

     # Per-stock: Individual Materials stock vs commodity basket
     path('api/mss-individual-stock-commodity-alignment/', views.mss_individual_stock_commodity_alignment, name='mss-individual-stock-commodity-alignment'),

     # Bulk: S&P 500 vs Technology sector
     path('api/mss-sp500-vs-tech-sector-analyzer/', views.mss_sp500_vs_tech_sector_analyzer, name='mss-sp500-vs-tech-sector-analyzer'),

     # ============================================================
     # URL ROUTES — Technology Subsector Analysis
     # ============================================================

     # Bulk: Compare all Technology subsectors
     path('api/mss-tech-subsector-bulk-analyzer/', views.mss_tech_subsector_bulk_analyzer, name='mss-tech-subsector-bulk-analyzer'),

     # Per-stock: Individual Tech stock vs its subsector peers
     path('api/mss-tech-stock-subsector-alignment/', views.mss_tech_stock_subsector_alignment, name='mss-tech-stock-subsector-alignment'),

     # ============================================================
     # URL ROUTE — Institutional vs Retail Analysis
     # ============================================================

     # Per-asset: Infer institutional vs retail influence on price action
     path('api/mss-institutional-vs-retail-analyzer/', views.mss_institutional_vs_retail_analyzer, name='mss-institutional-vs-retail-analyzer'),

     # Bulk: Deep analysis of any sector (health, drivers, opportunities)
     path('api/mss-sector-deep-dive-analyzer/', views.mss_sector_deep_dive_analyzer, name='mss-sector-deep-dive-analyzer'),

     # ============================================================
     # URL ROUTES — Assets of Interest + Stock Popularity
     # ============================================================

     # Toggle asset as 'of interest' for current trading day
     path('api/mss-toggle-asset-of-interest/', views.mss_toggle_asset_of_interest, name='mss-toggle-asset-of-interest'),

     # Get all assets marked for today
     path('api/mss-get-todays-assets/', views.mss_get_todays_assets, name='mss-get-todays-assets'),

     # Check if specific asset is saved today
     path('api/mss-check-asset-saved/', views.mss_check_asset_saved, name='mss-check-asset-saved'),

     # Analyze stock popularity using OpenAI
     path('api/mss-stock-popularity-analyzer/', views.mss_stock_popularity_analyzer, name='mss-stock-popularity-analyzer'),

     # Trade execution endpoints
     path('api/snowai-execute-trade-order-placement/', views.snowai_execute_trade_order_placement, name='snowai_execute_trade_order_placement'),
     path('api/snowai-close-trade-order-execution/', views.snowai_close_trade_order_execution, name='snowai_close_trade_order_execution'),
     path('api/snowai-fetch-trade-history-for-asset/', views.snowai_fetch_trade_history_for_asset, name='snowai_fetch_trade_history_for_asset'),
     path('api/snowai-fetch-overall-trading-performance/', views.snowai_fetch_overall_trading_performance, name='snowai_fetch_overall_trading_performance'),
     
     # Paper trading / backtest endpoints
     path('api/snowai-start-paper-trading-backtest/', views.snowai_start_paper_trading_backtest, name='snowai_start_paper_trading_backtest'),
     path('api/snowai-add-trade-to-backtest-session/', views.snowai_add_trade_to_backtest_session, name='snowai_add_trade_to_backtest_session'),
     path('api/snowai-complete-backtest-session/', views.snowai_complete_backtest_session, name='snowai_complete_backtest_session'),
     path('api/snowai-fetch-all-backtest-sessions/', views.snowai_fetch_all_backtest_sessions, name='snowai_fetch_all_backtest_sessions'),
     
     # Utility endpoints
     path('api/snowai-check-and-execute-stop-loss-take-profit/', views.snowai_check_and_execute_stop_loss_take_profit, name='snowai_check_and_execute_stop_loss_take_profit'),
     path('api/snowai-delete-trade-order/', views.snowai_delete_trade_order, name='snowai_delete_trade_order'),

     # Add the function from snowai_fetch_stock_info.py

     # Then add to urls.py:
     path('api/snowai-fetch-stock-info/', views.snowai_fetch_stock_info, name='snowai_fetch_stock_info'),


    # create appproprate urls.py here
    # path('test-async-backtest', views.test_async_backtest, name='test-async-backtest'),
    

    path('zinaida-feedback', views.zinaida_feedback_form, name='zinaida-feedback'),


    path('contact-us', views.contact_us, name='contact-us'),
    path('book-order', views.book_order, name='book-order'),
    path('api/register/', UserRegistrationView.as_view(), name='user-register'),
    path('api/login/', views.user_login, name='user-login'),
    path('api/csrf_token/', views.get_csrf_token, name='get_csrf_token'),
    # path('fetch_user_email', views.fetch_user_email, name='fetch_user_email')
]


