
# PROJE: Sepet aşamasındaki kullanıcılara ürün önerisinde bulunmak.
######################################################################

# Veri Seti Hikayesi

# Online Retail II isimli veri seti İngiltere merkezli online bir satış mağazasının 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını içeriyor.
# Bu şirketin ürün kataloğunda hediyelik eşyalar yer alıyor. Promosyon ürünleri olarak da düşünülebilir.
# Çoğu müşterisinin toptancı olduğu bilgisi de mevcut.

### Değişkenler ####
# InvoiceNo – Fatura Numarası
# Eğer bu kod C ile başlıyorsa işlemin iptal edildiğini ifade eder.
# StockCode – Ürün kodu Her bir ürün için eşsiz numara.
# Description – Ürün ismi Quantity – Ürün adedi
# Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate – Fatura tarihi UnitPrice – Fatura fiyatı (Sterlin)
# CustomerID – Eşsiz müşteri numarası Country – Ülke ismi

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)


pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

from google.colab import files
df_= files.upload()
#veri dosyası yükleme

d_f = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")
# 2010-2011 verileri seciliyor.

df = d_f.copy()
df.head()

####### VERİ ÖN İŞLEME ############

# outlier_tresholds bir değişkenin eşik degerini belirler.
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


#replace_with_tresholds belirlenen degerleri aykırı degerlerle değiştiriyor.
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


#retail_data_prep bizim kurallarımızdı.
# iadeleri cıkar 'c'
# ürün sayısı ve kazanc miktarı 0'dan fazla olsun
# eşik degerleri aykırı degerlerle yer değiştir
def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

# fonksiyonu uygula
df = retail_data_prep(df)


# ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)

# Germenay için ilgili işlemleri yapacagız.
df_fr = df[df['Country'] == "Germany"]

df_fr.head()

# istedigimiz tablo fatura numaralarına göre;
# faturada belirtilen ürün var mı yok mu
# varsa 1, yoksa 0

df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).head(50)

df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().head()
# her fatura için tüm ürünleri gösterir.
# unstack() pivotlamak.

df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().iloc[0:10, 0:5]
# Germany verileri üzerinde, ınvoice(fatura) ve (ürün acıklaması)description'a  göre grupla;
# gruplanan verilerin quantitylerini saydır. (toplam miktarı)
# iloc sayesinde 10 satır ve 5 sutun gösterim yapar.

df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:10, 0:5]
# fillna ile nan degerleri 0 ile doldur.

# asıl amacımız tabloda varsa 1 yoksa 0 degerleri ile doldurmak. kac tane oldugunu bulmak degil. yapacagımız işlem;
# tüm degerleri gez eger 0 dan büyükse 1 yaz degilse 0

df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).applymap(
    lambda x: 1 if x > 0 else 0).iloc[0:10, 0:5]



def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    
# if id: eger id girildiyse, stockCode'a göre
# eger id girilmediyse(else)(id=false) decription'a göre işlem yapar.

fr_inv_pro_df = create_invoice_product_df(df_fr, id=True)
# id True, stockCode'a göre gösterir.

fr_inv_pro_df.iloc[0:10,0:10]


# ID'leri verilen ürünlerin isimleri nelerdir?

# id'nin description'u.
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

check_id(df_fr, 21987)
# 21987 stockCode'lu ürün acıklaması (description) nedir?

check_id(df_fr, 23235)
# 23235 stockCode'lu ürün acıklaması (description) nedir?

check_id(df_fr, 22747)
# 22747 stockCode'lu ürün acıklaması (description) nedir?

# Birliktelik Kurallarının Çıkarılması

# Tüm olası ürün birlikteliklerinin olasılıkları

frequent_itemsets = apriori(fr_inv_pro_df, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False)

# Birliktelik kurallarının çıkarılması:

rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.sort_values("support", ascending=False).head()
rules.sort_values("lift", ascending=False).head(500)


# (antecedents	consequents) bu iki ürünün karsılastırılması alınır. support degerleri hesaplanır.

# support: İkisinin birlikte görülme olasılığı
# confidence: X alındığında Y alınma olasılığı.
# lift: X alındığında Y alınma olasılığı şu kadar kat artıyor.
# leverage: lifte benzer. supportu yüksek olan degerlere öncelik verme eğilimindedir. (cok kullanılmaz)
# conviction: Y olmadıgında X'in beklenen frekansı

# Çalışmanın Scriptini Hazırlama
#######################################

df_ = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

pd.set_option('display.max_columns', None)
from mlxtend.frequent_patterns import apriori, association_rules

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


#bu fonksiyonun görevi kuralları cıkarmak
def create_rules(dataframe, id=True, country="Germany"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)

    #apriori
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

df = retail_data_prep(df)
rules = create_rules(df)

df.head()

rules.head()

"""# Görev 4"""

# Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    #lifte göre sırala
    recommendation_list = []

    for i, product in sorted_rules["antecedents"].items():
      #antecedents içinde birden fazla ürün olabileceginden listeye cevirmeliyiz.
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))


    
    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
# çoklamaları ortadan kaldırır.

    return recommendation_list[:rec_count]
    # [:rec_count]  consequents'den kaç ürün sececegini belirtir.

arl_recommender(rules, 21987, 5)
# 21987 id li ürün için 5 tane öneri getir.

arl_recommender(rules, 23235, 5)
# 23235 id li ürün için 5 tane öneri getir.

arl_recommender(rules, 22747, 5)
# 22747 id li ürün için 5 tane öneri getir.

"""# Görev 5 """

#Tavsiye edilen ürünlerin isimleri

check_id(df,21124)

check_id(df,22029)

check_id(df,22423)

