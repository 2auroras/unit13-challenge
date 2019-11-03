import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

@pd.api.extensions.register_dataframe_accessor("tidy")


class Tidy:
    
    def __init__(self,pandas_obj,*target):
        
        self._obj = pandas_obj
        self.target = target
        
    
    
    def drop_lowvar(self):
        """
        Typical syntax df.tidy.drop_lowvar()
        Drops all columns whose values are the same, ie.zero variance.  
        No arguments needed or accepted.
        
        """
        dropped_columns = []
        count = 0
        
        for column in self._obj:
            if self._obj[column].nunique() == 1:
                dropped_columns.append(column)
                count += 1
                self._obj.drop([column], axis=1, inplace=True)
        
        if count > 0:
            print(f"The following {count} columns were dropped:\n-----------------\n{dropped_columns}")
        else:
            print("No columns were found with Zero variance.")
       
        
    def set_Xy(self,target):
        """
        Typical syntax X,y = df.tidy.encode_cats(). 
        Divdes the DataFrame into X(features) and y(target) DataFrames.   
        Must provide target as argument. Target can be a variable or a string.  
        Do not include brackets around target argument such as [column_name]
        
        """
        y = self._obj
        
        y = pd.DataFrame(y[target])
        
        X = self._obj
        
        X = pd.DataFrame(X.drop(columns=[target]))
        
        return X, y
    
    
    def encode_cats(self):
        """
        Typical syntax:  X = X.tidy.encode_cats().  
        Uses sklearn onehot encoding.  Identifies features that are categorical by looking for datatypes == object. 
        Ask user to confirm that none of the categories are ordinal (which should not be onehot encoded).  
        If the user identifies ordinal columns, they will be removed from the list of columns to be encoded.  
        NOTE: There could be cases of categorical data that this function will not catch.  Examine your data.
        
        """
        X = self._obj
        original_nominals = []
        count=0
        new_nominal_cat_names = []

        for column in self._obj.columns:
            if self._obj[column].dtypes==object:
                count+=1
                original_nominals.append(column)
        
        print(f"{count} categorial features were found (datatype == object):\n--------------\n{original_nominals}\n")
        
        answer = input("The above categorical features were found. Are any of them ordinal? (y/n)")
        
        if answer in ("y","n"):

                if answer == "y":
                    ordinals = input("Which features are ordinal?")
                    
                    original_nominals = [feature for feature in original_nominals if feature not in ordinals]
                    
                    for cat in original_nominals:
                        for value in self._obj[cat].unique().tolist():
                            new_nominal_cat_names.append(value)
                            
                    print(f"\n{ordinals} will not be onehot encoded")
                    
                    onehot = OneHotEncoder(dtype=np.int, sparse=True)
        
                    X_ohe=pd.DataFrame(onehot.fit_transform(X[original_nominals]).toarray(),columns=new_nominal_cat_names,index=X.index)
        
                    X = X.drop(columns=original_nominals)
        
                    X = pd.concat([X,X_ohe],join="outer",axis=1)
                    
                    print("\nEncoding completed.")
                    
                    return X
                
                    

                else:
                    for cat in original_nominals:
                        for value in self._obj[cat].unique().tolist():
                            new_nominal_cat_names.append(value)
                            
                    onehot = OneHotEncoder(dtype=np.int, sparse=True)
        
                    X_ohe=pd.DataFrame(onehot.fit_transform(X[original_nominals]).toarray(),
                                        columns=new_nominal_cat_names,index=X.index)
        
                    X = X.drop(columns=original_nominals)
        
                    X = pd.concat([X,X_ohe],join="outer",axis=1)
                    
                    print("\nEncoding completed.")
                    return X
                
                    
                    
        else:
            print("Answer must by y or n.\nRun function again.")