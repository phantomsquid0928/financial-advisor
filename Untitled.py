#!/usr/bin/env python
# coding: utf-8

# In[6]:


import PySimpleGUI as gui
import FinanceDataReader as fdr
#import selenium as se
import pandas as pd
import mplfinance as mpf
import matplotlib as mpl
import numpy as np
import xlsxwriter
import concurrent.futures as cf

#from matplotlib import use
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
#use('TkAgg')

class Trader:
    money = 0 #used money, important
    last = 0 #paid per stock
    invested = 0 #how much did u invested
    stock = 0#stock amount u bought, only handles 1 stock
    profit = 0
    total = 0
    
    now = 0 #this indicates stock's now price
    high = 0
    low = 0
    start = 0
    price = 0 #indicates which price was called on trade
    prev = 0
    def update(self, start, high, low, now) :
        self.prev = self.now
        self.now = now
        self.high = high
        self.low = low
        self.start = start
        self.total = self.profit + self.now * self.stock
    def trade(self, amount, mod, price = None) :
        if amount == 0 :
            return -1
        if mod == 1: #buy
            if (self.profit < self.now*amount) :
                left = self.profit - self.now*amount
                self.profit = 0
                self.money -= left
            else :
                self.profit -= self.now * amount
            self.invested += self.now * amount
            self.stock += amount
            self.last = self.invested / self.stock
            self.price = self.now
            self.total = self.profit + self.now * self.stock
        if mod == 2: #sell normal
            if (self.stock - amount < 0) : 
                return -1
            if (self.stock == 1) :
                self.profit += price
                self.price = price
                self.invested = 0
                self.stock = 0
                self.last = 0
                self.total = self.profit + self.now * self.stock
            elif (self.stock != 1):
                self.profit += price * amount
                self.invested -= self.last * amount
                self.stock -= amount
                if (self.stock == 0) :
                    self.last = 0
                self.price = price
                self.total = self.profit + self.now * self.stock
        if mod == 3: #sellall
            if (self.stock == 0) :
                return -1
            self.profit += price * self.stock
            self.invested = 0
            self.stock = 0
            self.last = 0
            self.price = price
            self.total = self.profit + self.now * self.stock
        if mod == 4: #runsell
            if self.stock - amount < 0 :
                return -1
            if (self.stock == 1) :
                self.profit += self.now
                self.invested = 0
                self.stock = 0
                self.last = 0
                self.price = self.now
                self.total = self.profit + self.now * self.stock
            elif (self.stock != 1):
                self.profit += self.now * amount
                self.invested -= self.last * amount
                self.stock -= amount
                if (self.stock == 0) :
                    self.last = 0
                self.price = self.now
                self.total = self.profit + self.now * self.stock
    def getStatus(self) :
        return [self.now, self.price, self.money, self.profit, self.last, self.invested, self.stock, self.total]
class SimulatedResults :
    def __init__(self, chart, name, start, end, ledge, result) :
        self.chart = chart
        self.name = name
        self.start = start
        self.end = end
        self.ledge = ledge
        self.result = result
    def __str__(self):
        return "[" + self.name + "] : " + str(self.result.invested) + " period: [" + str(self.start)+ "] ~ [" + str(self.end) + "]" + " [TOTAL]: " + str(self.result.total)
headings = ["  date  ", "   open   ", "   high   ",
            "  low  ", " close ", "   volume   ", "     change     "]
headings2 = [" name ", "   money   ", "  total  ", "stock", "  profit  "]
select_list = [[
    gui.Text("stock name: "),
    gui.In(size=(10, 1), enable_events=True, key="-stock-"),
    gui.Button("ok"),
    gui.In(size=(10, 1), enable_events=True, key="start_day"),
    gui.In(size=(10, 1), enable_events=True, key="end_day"),
    gui.Button("calc", disabled=True),
    gui.Text("select market: "),
    gui.Combo(['KOSPI', 'KOSDAQ', 'NYSE', 'NASDAQ', 'S&P500'], 
                enable_events=True, key='market_list'),
    gui.Button("calc_all"),
    gui.Button("show_result", disabled = True),
    gui.Button("export_excel")
],
[
    gui.ProgressBar(100, orientation='h', size = (70, 20), key = 'progbar', bar_color=['Skyblue', 'White']),
    gui.Text("(0 / 100) % prog", key="progtext")
],
[
   gui.Table(values=[], headings = headings, max_col_width = 30, 
             auto_size_columns=True, num_rows=20, row_height=25, key="stock_table") 
]]
layout = [[
    gui.Column(select_list), gui.VSeperator(),
    gui.Canvas(key="figCanvas", background_color='#000000')
]]

kp = []

window = gui.Window("stock", layout, finalize = True)
window2 = None
figure_agg = None
figure_agg2 = None
status = None
rank = []
ranks = []

def get_win2_layout() :
    layout2 = [[
    gui.Table(values=[], headings = headings2, max_col_width = 30, 
              auto_size_columns = True, num_rows = 20, row_height = 25, select_mode = gui.TABLE_SELECT_MODE_BROWSE,
              enable_events = True, key = "res_table"),
    gui.VSeperator(), 
    gui.Canvas(key="resCanvas")
]]
    return layout2

def draw_figure(canvas, figure) :
    figure_agg = FigureCanvasTkAgg(figure, canvas)
    figure_agg.draw()
    figure_agg.get_tk_widget().pack(side = 'right', fill='both', expand=1)
    return figure_agg
def draw(target, addplot = None, mod = 1) :
    fig = None
    axlist = None
    kwargs = dict(type = 'candle', mav = (3, 14, 30), volume = True, figratio = (18, 10), figscale = 2)
    
    mc = mpf.make_marketcolors(up='g', down='r', wick='black', volume='in')
    s = mpf.make_mpf_style(base_mpl_style='bmh', figcolor = 'black', marketcolors = mc)
    if (addplot != None) :
        fig, axlist = mpf.plot(target, **kwargs, style='yahoo', returnfig=True, addplot=addplot)
    else :
        fig, axlist = mpf.plot(target, **kwargs, style='yahoo', returnfig=True)
    if mod == 1:
        figure_agg = draw_figure(window["figCanvas"].TKCanvas, fig)
        return figure_agg
    else:
        figure_agg2 = draw_figure(window2["resCanvas"].TKCanvas, fig)
        return figure_agg2

def simulate1(info) :
    ledge = []
    first = 0
    chk2 = 0
    tr = Trader()
    for x in info:
        if first == 0 :
            start = info[x].loc["Open"]
            high = info[x].loc["High"]
            low = info[x].loc["Low"]
            now = info[x].loc["Close"]
            if (now == 0) : continue #trade stopped days
            tr.update(start, high, low, now)
            amount = 10
            tr.trade(amount, 1)
            first = 1
            #display(tr.getStatus(), 0)
            ledge.append([0, x, deepcopy(tr)])
            continue
        start = info[x].loc["Open"]
        high = info[x].loc["High"]
        low = info[x].loc["Low"]
        now = info[x].loc["Close"]
        tr.update(start, high, low, now)
        chker = 0
        '''if (tr.prev * 90 / 100 >= tr.now and chk2 <= 0) : #highly decreased, preventing
            chk2 = 1
            amount = tr.stock - tr.stock // 2
            tr.trade(amount, 4)
            display(tr.getStatus(), "s#1, highly decrease in 1 day, sell off")
            ledge.append([4, x, deepcopy(tr)])
            continue '''
        if (tr.last * 80 / 100 >= tr.now and chk2 <= 0) : #10% down ,runsell
            chk2 = 1
            amount = tr.stock - tr.stock // 2
            tr.trade(amount, 4)
            #display(tr.getStatus(), "s5")
            ledge.append([5, x, deepcopy(tr)])
            continue
        '''if (tr.last * 95 / 100 >= tr.now) : #5%down
            amount = 20
            chk2 -= 1
            tr.trade(amount, 1)
            display(tr.getStatus(), "b7")
            ledge.append([6, x, deepcopy(tr)])
            continue'''
        if (tr.last > tr.now or tr.last == 0) :
            amount = 10
            chk2 -= 1
            tr.trade(amount, 1)
            #display(tr.getStatus(), "b8")
            ledge.append([7, x, deepcopy(tr)])
            continue
        if (tr.last < tr.now) :
            if (tr.last * 105 / 100 <= tr.high and chker <= 0) : #10%up
                chker = 3
                amount = tr.stock - tr.stock // 2
                #display(tr.getStatus(), "s3")
                tr.trade(amount, 2, tr.last * 1.05)
            if (tr.last * 110 / 100 <= tr.high) : #30% up
                chker = 2
                amount = tr.stock - tr.stock * 2 // 5
                #display(tr.getStatus(), "s2")
                tr.trade(amount, 2, tr.last * 1.1)
            if (tr.last * 115 / 100 <= tr.high) : #50% up, sellall
                chker = 1
                #display(tr.getStatus(), "s1")
                tr.trade(1, 3, tr.last * 1.2)
            if (chker != 0) :
                ledge.append([chker, x, deepcopy(tr)])
                continue
        ledge.append([8, x, deepcopy(tr)])
    #tr.trade(1, 3) #u can change this if u change money check to total check
    #ledge.append(["finalized", tr])
    return ledge, tr

while True:
    t_win, event, values = gui.read_all_windows()
    if event == "Exit" or event == gui.WIN_CLOSED:
        if t_win == window2 :
            if (figure_agg2) :
                figure_agg2.get_tk_widget().forget()
            t_win.close()
            window2 = None
        elif t_win == window :
            t_win.close()
            break
    if event == "ok":
        status = "ok"
        try:
            target = values["-stock-"]
            if (target == None) :
                pass
            kp = fdr.DataReader(target, values['start_day'], values['end_day'])
            #window.Refresh()
            if (figure_agg) :
                figure_agg.get_tk_widget().forget()
                #window["figCanvas"].update()
            if (figure_agg2) :
                figure_agg2.get_tk_widget().forget()
            figure_agg = draw(kp, None, 1)
            info = kp.transpose()
            temp = []
            value = []
            for x in info:
                temp = [x]
                for t in info[x].values.tolist() :
                    temp.append(t)
                value.append(temp)
            window["stock_table"].update(values=value)
            window["calc"].update(disabled=False)
        except:
            window["-stock-"].update("does not exists")
            window["calc"].update(disabled=True)
    if event == "calc" :
        rank = []
        status = "calc"
        if (figure_agg) :
            figure_agg.get_tk_widget().forget()
        info = kp.transpose()
        ledge, result = simulate1(info)
        rank.append(SimulatedResults(kp, values["-stock-"], values['start_day'], 
                                     values['end_day'], ledge, result))

        addplot = []
        scatter1 = []
        scatter2 = []
        chker = 0
        chker1 = 0
        last = []
        profit = []
        for hist in ledge:
            if (hist[0] > 0 and hist[0] <= 5) :
                chker = 1
                scatter1.append(hist[2].now * 1.1)
                scatter2.append(np.nan)
            elif (hist[0] >= 6 and hist[0] <= 7) :
                chker1 = 1
                scatter1.append(np.nan)
                scatter2.append(hist[2].now * 0.9)
            else :
                scatter1.append(np.nan)
                scatter2.append(np.nan)
            last.append(hist[2].last)
            profit.append(hist[2].total)
        if (chker == 0 and chker1 == 0) :
            window["-stock-"].update("does not exists")
            continue
        if (chker == 0) :
            addplot = [mpf.make_addplot(scatter2, type = 'scatter', markersize=10, marker='^'),
                  mpf.make_addplot(last, secondary_y = False, markersize = 5, color = 'b'),
                  mpf.make_addplot(profit, title="profit", markersize = 5, color = 'y')]
        else :
            addplot = [mpf.make_addplot(scatter1, type = 'scatter', markersize=10, marker='v'),
                  mpf.make_addplot(scatter2, type = 'scatter', markersize=10, marker='^'),
                  mpf.make_addplot(last, secondary_y = False, markersize = 5, color = 'b'),
                  mpf.make_addplot(profit, title="profit", markersize = 5, color = 'y')]
        figure_agg = draw(kp, addplot, 1)
        value2 = []
        value2.append([values["-stock-"], result.money, result.total, result.stock, result.total - result.money])
        window2 = gui.Window("stock", get_win2_layout(), finalize = True)
       
        window2["res_table"].update(values = value2)
    if event == "calc_all" :
        status = "calc_all"
        window["calc"].update(disabled=True)
        window["show_result"].update(disabled = False)
        if (figure_agg2) :
            figure_agg2.get_tk_widget().forget()
        market = values['market_list']
        if (market == '') :
            window["-stock-"].update("plz select combo")
            continue

        results = []
        #display(market)
        targets = fdr.StockListing(market)

        loop = 0
        val = 0
        k = len(targets["Symbol"])
        for t in targets["Symbol"] :
            val = val + 100 / (k - 1)
            
            window['progbar'].update_bar(val)
            window['progtext'].update("(" + str(val) + "/ 100 ) prog")
            #display(t)
            try :
                cur = fdr.DataReader(t, values['start_day'], values['end_day'])
            except:
                #display("sth went wrong, passing")
                continue
            #display([t, "is under calculation"])
            executor = cf.ThreadPoolExecutor()
            future = executor.submit(simulate1, cur.transpose())
            ledge, result = future.result()
            #ledge, result = simulate1(cur.transpose())
            results.append(SimulatedResults(cur, t, values['start_day'], values['end_day'], ledge, result))
            loop = loop + 1
        rank = sorted(results, key=lambda x: x.result.total - x.result.money)
        ranks.append(rank)
        window2 = gui.Window("stock", get_win2_layout(), finalize = True)
        value = []
        for x in rank :
            print(x)
            value.append([x.name, x.result.money, x.result.total, x.result.stock, x.result.total - x.result.money])
        
        window2["res_table"].update(values = value)
    if event == "res_table" :
        if (figure_agg2) :
            figure_agg2.get_tk_widget().forget()

        data = rank[values["res_table"][0]]
        #display(data.name)
        ledge = data.ledge
        
        addplot = []
        scatter1 = []
        scatter2 = []
        chker = 0
        chker1 = 0
        last = []
        profit = []
        for hist in ledge:
            if (hist[0] > 0 and hist[0] <= 5) :
                chker = 1
                scatter1.append(hist[2].now * 1.1)
                scatter2.append(np.nan)
            elif (hist[0] >= 6 and hist[0] <= 7 or hist[0] == 0) :
                chker1 = 1
                scatter1.append(np.nan)
                scatter2.append(hist[2].now * 0.9)
            else :
                scatter1.append(np.nan)
                scatter2.append(np.nan)
            last.append(hist[2].last)
            profit.append(hist[2].total)
        
        if (chker == 0 and chker1 == 0) :
            window["-stock-"].update("does not exists")
            continue
        if (chker == 0) :
            addplot = [mpf.make_addplot(scatter2, type = 'scatter', markersize=10, marker='^'),
                  mpf.make_addplot(last, secondary_y = False, markersize = 5, color = 'b'),
                  mpf.make_addplot(profit, markersize = 5, color = 'y')]
        else :
            addplot = [mpf.make_addplot(scatter1, type = 'scatter', markersize=10, marker='v'),
                  mpf.make_addplot(scatter2, type = 'scatter', markersize=10, marker='^'),
                  mpf.make_addplot(last, secondary_y = False, markersize = 5, color = 'b'),
                  mpf.make_addplot(profit, markersize = 5, color = 'y')]
        
        figure_agg2 = draw(data.chart, addplot, 2)
    if event == "show_result" :
        window["show_result"].update(disabled = True)
        fig = mpl.pyplot.figure(figsize = (30, 30))
        dim3 = fig.add_subplot(projection = '3d')
        dim3.set_title("PROFIT RANK")
        dim3.set_xlabel("profit ($ or \)")
        dim3.set_ylabel("period (days)")
        dim3.set_zlabel("money spent")
        
        figure_agg = draw_figure(window['figCanvas'].TKCanvas, fig)
        for selected_rank in ranks:
            for t in selected_rank :
                start = datetime.strptime(t.start, "%Y-%m-%d")
                end = datetime.strptime(t.end, "%Y-%m-%d")
                diff = (start - end).days
                dim3.scatter(t.result.total - t.result.money, diff, t.result.money)
            best = selected_rank[len(selected_rank) - 1]
        
        figure_agg = draw_figure(window['figCanvas'].TKCanvas, fig)
        print(best)
        ranks = []
    if event == "export_excel" :
        #display(rank)
        temp = []
        header = ['day', 'trade_num', 'open', 'high', 'low', 'close', 'price', 'money_spent',
                  'profit', 'avg', 'invested', 'stock', 'total', 'pure_profit']
        for x in rank[0].ledge :
            if (x[0] == 8) :
                temp.append([x[1], x[0], x[2].start, x[2].high, x[2].low, x[2].now, 0, 
                         x[2].money, x[2].profit, x[2].last, x[2].invested, x[2].stock, x[2].total])
            else :
                temp.append([x[1], x[0], x[2].start, x[2].high, x[2].low, x[2].now, x[2].price, 
                         x[2].money, x[2].profit, x[2].last, x[2].invested, x[2].stock, x[2].total, x[2].total - x[2].money])
        data = pd.DataFrame(temp, columns= header)
        
        filename = rank[0].name + "(" + str(rank[0].start) + '~' + str(rank[0].end) + ")" + 'res.xlsx'
        data.to_excel(filename, 'sheet1', index=False, engine = 'xlsxwriter')
window.close()


