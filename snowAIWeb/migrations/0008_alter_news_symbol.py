# Generated by Django 4.2.4 on 2024-02-14 12:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("snowAIWeb", "0007_news_user_email"),
    ]

    operations = [
        migrations.AlterField(
            model_name="news",
            name="symbol",
            field=models.CharField(max_length=100),
        ),
    ]
