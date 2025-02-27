# Generated by Django 4.2.4 on 2025-02-28 04:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("snowAIWeb", "0038_backtestresult_backtest_model"),
    ]

    operations = [
        migrations.CreateModel(
            name="IdeaModel",
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
                ("idea_category", models.CharField(max_length=255)),
                ("idea_text", models.TextField()),
                (
                    "idea_tracker",
                    models.CharField(
                        choices=[
                            ("Pending", "Pending"),
                            ("In Progress", "In Progress"),
                            ("Completed", "Completed"),
                        ],
                        max_length=50,
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
