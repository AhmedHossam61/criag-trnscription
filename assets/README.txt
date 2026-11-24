Art for the lore site.

VAKT_Arcane_Heraldic_Crest.png
    The party emblem (page header, welcome screen, browser-tab icon),
    set by `emblem` under [party] in config/rag.toml.

One folder per character, named after them (Kael, Vaelor, Thalion, Arnok,
Hasaneen, Ronaldo Raddis the Third; a nickname like "Raddis" works too):
    portrait.png    shown on their page and in the party list. Without a file
                    called portrait, the folder's first image is used.
                    .png .jpg .webp all work; square-ish images crop best.
    anything else   free to keep here (more art, maps, tokens).

comics/ -- one folder per comic, named after it (comics/Two Minutes/):
    01.jpeg, 02.jpeg, ...   its pages, read in number order (2 before 10)
    comic.toml              optional: title, summary, date, characters

Changes show on the next page reload; no restart needed.
