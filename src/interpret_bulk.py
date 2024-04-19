import argparse
import interpret_models

if __name__ == '__main__':
    
    temp = [900, 1100, 1300, 1500, 1700, 1900]
    
    for num in temp:
        args = argparse.Namespace(model='dt', comps_1='30', comps_2='4', input=f'lichess-{num}-{num+200}-25k', 
                                  plot='all', suffix=num, plot_type=None)
        interpret_models.main(args)