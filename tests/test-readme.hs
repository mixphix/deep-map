{-# LANGUAGE TemplateHaskell #-}

module Main where

import Data.Foldable (fold)
import Data.Map.Deep
import Data.Map.Strict (Map)
import Data.Semigroup (Sum (..))
import Data.Text (Text)
import Data.Time.Compat
import Hedgehog
import Hedgehog.Main (defaultMain)

newtype OrderID = OrderID Int deriving (Eq, Ord, Show)

newtype CustomerID = CustomerID Text deriving (Eq, Ord, Show)

type Price = Sum Double

type Table = DeepMap '[Day, OrderID, CustomerID] Price

table :: Table
table =
  fromList3
    [ (YearMonthDay 2021 1 1, OrderID 1, CustomerID "Melanie", Sum 13.12),
      (YearMonthDay 2021 1 1, OrderID 2, CustomerID "Sock", Sum 4.20),
      (YearMonthDay 2021 1 2, OrderID 3, CustomerID "Sock", Sum 69.69),
      (YearMonthDay 2021 1 2, OrderID 4, CustomerID "Fiona", Sum 5.00)
    ]

totalSales :: Table -> Price
totalSales = fold

prop_totalsales :: Property
prop_totalsales = property $ totalSales table === Sum (13.12 + 4.20 + 69.69 + 5.00)

dailySales :: Table -> Map Day Price
dailySales = toMap . foldMapWithKey3 (\d _o _c p -> d @| p)

dailyCustomers :: Table -> Map Day [CustomerID]
dailyCustomers = toMap . foldMapWithKey3 (\d _o c _p -> d @| [c])

totalPerCustomer :: Table -> DeepMap '[CustomerID] Price
totalPerCustomer = foldShallow . foldShallow

customerSaleDates :: Table -> Map CustomerID [Day]
customerSaleDates = toMap . foldMapWithKey3 (\d _o c _p -> c @| [d])

orderTotalPerCustomer :: Table -> DeepMap '[CustomerID, OrderID] Price
orderTotalPerCustomer = invertKeys . foldShallow

sockTotal :: Table -> Double
sockTotal t = getSum $ totalPerCustomer t @!| CustomerID "Sock"

prop_socktotal :: Property
prop_socktotal = property $ sockTotal table === (69.69 + 4.20)

main :: IO ()
main = print table >> defaultMain [checkParallel $$(discover)]
