# Generated by Django 4.2.4 on 2024-04-07 10:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("snowAIWeb", "0012_alter_bookorder_interested_product"),
    ]

    operations = [
        migrations.AddField(
            model_name="bookorder",
            name="number_of_units",
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name="bookorder",
            name="phone_number",
            field=models.IntegerField(default=0),
        ),
    ]