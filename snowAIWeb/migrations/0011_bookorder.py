# Generated by Django 4.2.4 on 2024-04-04 21:55

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("snowAIWeb", "0010_contactus"),
    ]

    operations = [
        migrations.CreateModel(
            name="BookOrder",
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
                ("first_name", models.CharField(max_length=100)),
                ("last_name", models.CharField(max_length=100)),
                ("interested_product", models.TextField(max_length=100)),
                ("email", models.EmailField(max_length=254)),
            ],
        ),
    ]
