import torch

def collate_fn(batch):
    out = dict()
    out["S"] = {
        "item_id": torch.cat([torch.tensor(x["S"]["item_id"], dtype=torch.long) for x in batch]),
        "timestamps": torch.cat([torch.tensor(x["S"]["timestamps"], dtype=torch.long) for x in batch]),
        "lengths": torch.tensor([x["S"]["length"] for x in batch], dtype=torch.long)
    }
    out["NS"] = {
        "sparse_features" : {
            "uid" : torch.tensor([x["NS"]["sparse_features"]["uid"] for x in batch], dtype=torch.long),
            "item_id" : torch.tensor([x["NS"]["sparse_features"]["item_id"] for x in batch], dtype=torch.long)
        },
        "timestamp": torch.tensor([x["NS"]["timestamp"] for x in batch], dtype=torch.long),
        "dense_features" : torch.cat([torch.tensor(x["NS"]["dense_features"]).unsqueeze(0) for x in batch], dim=0),
        "multivalent_features" : {
            "album_ids" : {
                "values" : torch.cat([torch.tensor(x["NS"]["multivalent_features"]["album_ids"]["values"], dtype=torch.long) for x in batch]),
                "lengths" : torch.tensor([x["NS"]["multivalent_features"]["album_ids"]["length"] for x in batch])
            },
            "artist_ids" : {
                "values" : torch.cat([torch.tensor(x["NS"]["multivalent_features"]["artist_ids"]["values"], dtype=torch.long) for x in batch]),
                "lengths" : torch.tensor([x["NS"]["multivalent_features"]["artist_ids"]["length"] for x in batch], dtype=torch.long)
            }
        }
    }
    out["targets"] = {
        "is_like" : torch.tensor([x["targets"]["is_like"] for x in batch]),
        "is_full_play" : torch.tensor([x["targets"]["is_full_play"] for x in batch])
    }
    return out