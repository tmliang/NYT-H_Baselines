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
        </td><td> CNN+ATT </td><td> 60.272 </td><td> 38.527 </td><td> 44.762 </td>
    </tr>
    <tr>
        <td rowspan="3"> Bag2Bag </td><td> PCNN+ATT </td><td> 60.272 </td><td> 38.527 </td><td> 44.762 </td>
    </tr>
    <tr>
        </td><td> BiGRU+ATT </td><td> 60.272 </td><td> 38.527 </td><td> 44.762 </td>
    </tr>
</table>

