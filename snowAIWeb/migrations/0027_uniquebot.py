# Generated by Django 4.2.4 on 2024-06-13 05:49

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("snowAIWeb", "0026_alter_trademodel_order_ticket"),
    ]

    operations = [
        migrations.CreateModel(
            name="uniqueBot",
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
                ("model_id", models.IntegerField()),
                ("order_ticket", models.TextField()),
                ("asset", models.CharField(max_length=20, null=True)),
                ("bot_id", models.TextField()),
            ],
        ),
    ]
