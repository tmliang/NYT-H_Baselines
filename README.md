# NYT-H_Baselines
Baselines of NYT-H Dataset.

## Data
Download the dataset from [NYT-H](https://github.com/Spico197/NYT-H) and the Glove embedding from [here](http://nlp.stanford.edu/data/glove.6B.zip), unzip them in `data` and `glove` folder respectively.

## Train and Test
```
CUDA_VISIBLE_DEVICES=0 python main.py --encoder PCNN --lr 0.5 --batch_size 100
```

## Experimental Result

<table>
    <tr>
        <th> Track </th><th> Model </th><th> Precsion </th> <th> Recall </th> <th> F1-Score </th>
    </tr>
    <tr>
        <td rowspan="3"> Bag2Bag </td><td> CNN+ATT </td><td align="center"> 52.308 </td><td align="center"> 34.460 </td><td align="center"> 38.906 </td>
    </tr>
    <tr>
        <td> PCNN+ATT </td><td align="center"> 61.160 </td><td align="center"> 38.547 </td><td align="center"> 45.108 </td>
    </tr>
    <tr>
        </td><td> BiGRU+ATT </td><td align="center"> 51.658 </td><td align="center"> 27.471 </td><td align="center"> 31.608 </td>
    </tr>
</table>

