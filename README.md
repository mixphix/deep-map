# deep-map

[![ci](https://img.shields.io/github/workflow/status/cigsender/deep-map/Haskell-CI)](https://github.com/cigsender/deep-map/actions/workflows/ci.yml)
[![Hackage](https://img.shields.io/hackage/v/deep-map?color=purple)](https://hackage.haskell.org/package/deep-map)
[![license](https://img.shields.io/github/license/cigsender/deep-map?color=purple)]()

A `DeepMap` is a map that has several layers of keys.

```hs
type DeepMap :: [Type] -> Type -> Type
data DeepMap ks v where
    Bare :: v -> DeepMap '[] v
    Nest :: Map k (DeepMap ks v) -> DeepMap (k ': ks) v
```

For a given `(k ': ks) :: [Type]`, the type `DeepMap (k ': ks) v` is isomorphic to lists of the form `[(k, k0, .., kn, v)]` where `ks = '[k0, ..., kn]`, but with better performance.

## Example

Say you have a table of order IDs, dates, customer IDs, and the total price for that order; and you'd like to track some statistics.

```hs
newtype OrderID = OrderID Int
newtype CustomerID = CustomerID Text
newtype Price = Sum Double

type Table = DeepMap '[Day, OrderID, CustomerID] Price

table :: Table
table =
  fromList3
    [ (YearMonthDay 2021 1 1, OrderID 1, CustomerID "Melanie", Sum 13.12)
    , (YearMonthDay 2021 1 1, OrderID 2, CustomerID "Sock", Sum 4.20)
    , (YearMonthDay 2021 1 2, OrderID 3, CustomerID "Sock", Sum 69.69)
    , (YearMonthDay 2021 1 2, OrderID 4, CustomerID "Fiona", Sum 5.00)
    ]

totalSales :: Table -> Price
totalSales = fold

-- How much did customers spend on a given day?
-- (note: use a DeepMap accumulator to fold within the Semigroup)
dailySales :: Table -> Map Day Price
dailySales = toMap . foldMapWithKey3 (\d _o _c p -> d @| p)

-- Who purchased something on which day?
dailyCustomers :: Table -> Map Day [CustomerID]
dailyCustomers = toMap . foldMapWithKey3 (\d _o c _p -> d @| [c])

-- How much has a customer paid throughout history?
totalPerCustomer :: Table -> DeepMap '[CustomerID] Price
totalPerCustomer = foldShallow . foldShallow
{- = foldMapWithKey3 (\_d _o c p -> c @| p) -}

-- What days did each customer purchase something?
customerSaleDates :: Table -> Map CustomerID [Day]
customerSaleDates = toMap . foldMapWithKey3 (\d _o c _p -> c @| [d])
{- = toMap . Map.foldMapWithKey (\d cs -> foldMap (@| [d]) cs) . dailyCustomers -}

-- How much did a customer spend for each order?
-- Useful if e.g. several customers could chip into the same order.
orderTotalPerCustomer :: Table -> DeepMap '[CustomerID, OrderID] Price
orderTotalPerCustomer = invertKeys . foldShallow

-- Using (@!) will NOT throw an error!
-- It's the infix version of `findWithDefault mempty`.
sockTotal :: Table -> Double
sockTotal t = getSum $ totalPerCustomer t @!| CustomerID "Sock"
```

## Shallow & deep functions

You might have noticed the difference between `fold` and `foldShallow` in the above example:

```hs
fold :: (Monoid v) => DeepMap (k ': ks) v -> v
foldShallow :: (Monoid (DeepMap ks v)) => DeepMap (k ': ks) v -> DeepMap ks v
```

Here are a few other pairs of similar functions, that operate at different depths:

```hs
fmap :: (v -> w) -> DeepMap ks v -> DeepMap ks w
mapShallow :: (DeepMap ks v -> DeepMap ls w) -> DeepMap (k ': ks) v -> DeepMap (k ': ls) w

traverse :: (Applicative f) => (v -> f w) -> DeepMap ks v -> f (DeepMap ks w)
traverseShallow :: (Applicative f) => (DeepMap ks v -> f (DeepMap ls w)) -> DeepMap (k ': ks) v -> f (DeepMap (k ': ls) w)

mapMaybe :: (v -> Maybe w) -> DeepMap (k ': ks) v -> DeepMap (k ': ks) w
mapShallowMaybe :: (DeepMap ks v -> Maybe (DeepMap ls w)) -> DeepMap (k ': ks) v -> DeepMap (k ': ls) w

mapEither ::
  (v -> Either w x) ->
  DeepMap (k ': ks) v ->
  (DeepMap (k ': ks) w, DeepMap (k ': ks) x)
mapShallowEither ::
  (DeepMap ks v -> Either (DeepMap ls w) (DeepMap ms x)) ->
  DeepMap (k ': ks) v ->
  (DeepMap (k ': ls) w, DeepMap (k ': ms) x)
```
