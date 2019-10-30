

# Challenge Set 9

## Part I: W3Schools SQL Lab 

*Introductory level SQL*

--

This challenge uses the [W3Schools SQL playground](http://www.w3schools.com/sql/trysql.asp?filename=trysql_select_all). Please add solutions to this markdown file and submit.

1. Which customers are from the UK?

1. ````sql
   SELECT * FROM Customers
   WHERE Country='UK';
   ````

2. What is the name of the customer who has the most orders?

   **Ernst Handel**

   ````sql
   SELECT CustomerID, COUNT(OrderID) as number_of_orders
   FROM Orders
   GROUP BY CustomerID
   ORDER BY number_of_orders DESC;
   ````

3. Which supplier has the highest average product price?

   **Supplier 18: Aux joyeux ecclésiastiques**

   ````sql
   SELECT SupplierID, AVG(Price) as avg_price
   FROM Products
   GROUP BY SupplierID
   ORDER BY avg_price desc;
   ````

4. How many different countries are all the customers from? (*Hint:* consider [DISTINCT](http://www.w3schools.com/sql/sql_distinct.asp).)

   **21**

   ````sql
   SELECT COUNT(DISTINCT(Country))
   FROM Customers
   ````

5. What category appears in the most orders?

   **Category 4: Dairy Products**

   ````sql
   SELECT Products.CategoryID, COUNT(Products.CategoryID)
   FROM OrderDetails
   LEFT JOIN Products ON OrderDetails.ProductID = Products.ProductID
   GROUP BY CategoryID;
   ````

6. What was the total cost for each order?

   ````sql
   SELECT OrderDetails.OrderID, SUM(OrderDetails.Quantity * Products.Price) AS price_of_order
   FROM OrderDetails
   LEFT JOIN Products ON OrderDetails.ProductID = Products.ProductID
   GROUP BY OrderID;
   ````

7. Which employee made the most sales (by total price)?

   **Employee 4: Margaret Peacock**

   ````sql
   CREATE VIEW price_of_orders AS
   SELECT OrderDetails.OrderID, SUM(OrderDetails.Quantity * Products.Price) AS price_of_order
   FROM OrderDetails
   LEFT JOIN Products ON OrderDetails.ProductID = Products.ProductID
   GROUP BY OrderID;
   ````

   ````sql
   SELECT EmployeeID, SUM(price_of_orders.price_of_order)
   FROM Orders
   LEFT JOIN price_of_orders ON Orders.OrderID = price_of_orders.OrderID
   GROUP BY EmployeeID;
   ````

8. Which employees have BS degrees? (*Hint:* look at the [LIKE](http://www.w3schools.com/sql/sql_like.asp) operator.)

   **Employees 3 and 5: Janet Leverling and Steven Buchanan**

   ````sql
   SELECT *
   FROM Employees
   WHERE Notes LIKE '%BS%';
   ````

9. Which supplier of three or more products has the highest average product price? (*Hint:* look at the [HAVING](http://www.w3schools.com/sql/sql_having.asp) operator.)

   **Supplier 4: Tokyo Traders**

   ````sql
   SELECT SupplierID, AVG(Price) as avg_price
   FROM Products
   GROUP BY SupplierID
   HAVING COUNT(SupplierID) >= 3
   ORDER BY avg_price desc;
   ````
