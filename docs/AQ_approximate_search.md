<details><summary>Searching with the AQ approximation</summary>

AQ search with QINCo re-ranking can be performed with:

```
python -u search_2stage.py  \
  --model models/bigann_8x8_L2.pt --db bigann1M
```

Which yields [this output](https://gist.github.com/mdouze/7a3b3e5431a9b36d392969cd506f34cf)
which corresponds to the first column of Table 4 in the paper. 

Note that this is based on decompressed vectors, there is no real fast search implementation.
Stay tuned: this is implemented in the IVF experiments...

</details>
