
void OnTick()
  {
   
   int ticket;
   
   double lotSize = 0.01;
   int slippage = 3;
   string currentOrder = "";
   for (int i = OrdersTotal() - 1; i >= 0; i--) {
       if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) {
           int orderType = OrderType();
           Print("Order ", OrderTicket(), " is a ", (orderType == OP_BUY ? "buy" : "sell"), " order.");
           currentOrder = orderType == OP_BUY ? "buy" : "sell";
       } else {
           Print("Error selecting order. Error code: ", GetLastError());
       }
   }
   
   string cookie = NULL;
   string headers;
   char post[];
   char result[];
   int res;

   //--- Reset the last error code
   ResetLastError();
   
   //--- Timeout below 1000 (1 sec.) is not enough for slow Internet connection
   int timeout = 10000;
   string symbol = Symbol();
   string url = "https://backend-production-c0ab.up.railway.app/api-call/" + symbol;
   res = WebRequest("GET", url, cookie, NULL, timeout, post, 0, result, headers);

   //--- Checking errors
   if(res == -1)
   {
      Print(StringFormat("Error in WebRequest. Error code = %i, url = %s", GetLastError(), url));
   }
   else if (res == 200)
   {
      // Success, return result as a string...
      Print(CharArrayToString(result));
      string jsonResponse = CharArrayToString(result);
      string message;
      if (StringToJSON(jsonResponse, message))
         {
             Print("JSON Response: ", jsonResponse);
             Print("Message: ", message);
             Print(typename(message));
             //--- If message == 1: Buy, else if message == -1: Sell, else: do nothing.
             if (message == "1" &&  currentOrder != "buy") {
               ///--- if open position == sell() close()
               if (currentOrder == "sell") {
                  ClosePositions();
               }
               ticket = OrderSend(Symbol(), OP_BUY, lotSize, Ask, slippage, 0, 0, "", NULL, 0, Green);
           
             }
             
             else if (message == "-1"  &&  currentOrder != "sell") {
             ///--- if open position == buy() close()
                  if (currentOrder == "buy") {
                  ClosePositions();
               }
               ticket = OrderSend(Symbol(), OP_SELL, lotSize, Ask, slippage, 0, 0, "", NULL, 0, Green);
             }
             else {
               Print("No trade necessary to execute");
             }
         }
       else
         {
             Print("JSON parsing failed. Response: ", jsonResponse);
         }
   }
   else
   {
      Print(StringFormat("WebRequest to %s failed. Error code = %i", url, res));
   }
   
   Sleep(300000);
   
  }
//+------------------------------------------------------------------+


bool StringToJSON(string json, string &message)
{
    int messageStart = StringFind(json, "\"message\":");
    if (messageStart != -1)
    {
        int messageValueStart = StringFind(json, "\"", messageStart + 10);
        if (messageValueStart != -1)
        {
            int messageValueEnd = StringFind(json, "\"", messageValueStart + 1);
            if (messageValueEnd != -1)
            {
                message = StringSubstr(json, messageValueStart + 1, messageValueEnd - messageValueStart - 1);
                return true;
            }
        }
    }
    
    return false;
    
}


void ClosePositions() {
    int totalOrders = OrdersTotal();
    for (int i = totalOrders - 1; i >= 0; i--) {
        if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) {
            if (OrderClose(OrderTicket(), OrderLots(), Bid, 3, Red)) {
                Print("Order closed successfully. Ticket number: ", OrderTicket());
            } else {
                Print("Error closing order. Error code: ", GetLastError());
            }
        } else {
            Print("Error selecting order. Error code: ", GetLastError());
        }
        
    }
}
