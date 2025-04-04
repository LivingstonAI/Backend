# Generated by Django 4.2.4 on 2024-12-17 09:27

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("snowAIWeb", "0031_alertbot"),
    ]

    operations = [
        migrations.CreateModel(
            name="Account",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("account_name", models.CharField(max_length=100, unique=True)),
                ("main_assets", models.CharField(max_length=255)),
                ("initial_capital", models.FloatField()),
            ],
        ),
        migrations.CreateModel(
            name="AccountTrades",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("asset", models.CharField(max_length=100)),
                ("order_type", models.CharField(max_length=50)),
                ("strategy", models.CharField(max_length=100)),
                ("day_of_week_entered", models.CharField(max_length=10)),
                (
                    "day_of_week_closed",
                    models.CharField(blank=True, max_length=10, null=True),
                ),
                ("trading_session_entered", models.CharField(max_length=50)),
                (
                    "trading_session_closed",
                    models.CharField(blank=True, max_length=50, null=True),
                ),
                ("outcome", models.CharField(max_length=10)),
                ("amount", models.FloatField()),
                ("emotional_bias", models.TextField(blank=True, null=True)),
                ("reflection", models.TextField(blank=True, null=True)),
                (
                    "account",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="trades",
                        to="snowAIWeb.account",
                    ),
                ),
            ],
        ),
    ]
