// Edit confirmed public information here. Empty fields are intentionally hidden.
window.RFWS_SITE = {
  name: "RFW-S",
  fullName: "RobotFlyWheel-Soft",
  title: "RFW-S: Everyday Deformable Objects with Independent Simulation Verification",
  // Latest user-supplied order, affiliations and markers: video v016 credits.json.
  authors: [
  {
    "name": "Jiarui Guo",
    "affiliations": [
      "1",
      "2"
    ],
    "markers": [
      "*"
    ]
  },
  {
    "name": "Saining Zhang",
    "affiliations": [
      "2",
      "3"
    ],
    "markers": [
      "*",
      "†"
    ]
  },
  {
    "name": "Jiaying Wan",
    "affiliations": [
      "2"
    ],
    "markers": []
  },
  {
    "name": "Jiawei Weng",
    "affiliations": [
      "1"
    ],
    "markers": []
  },
  {
    "name": "Junhao Chen",
    "affiliations": [
      "2"
    ],
    "markers": []
  },
  {
    "name": "Rui Zhang",
    "affiliations": [
      "1",
      "2"
    ],
    "markers": []
  },
  {
    "name": "Yuanbo Lyu",
    "affiliations": [
      "1"
    ],
    "markers": []
  },
  {
    "name": "Zhiheng Lee",
    "affiliations": [
      "2"
    ],
    "markers": []
  },
  {
    "name": "Peishuo Li",
    "affiliations": [
      "3"
    ],
    "markers": []
  },
  {
    "name": "Jingrun Du",
    "affiliations": [
      "1",
      "2"
    ],
    "markers": []
  },
  {
    "name": "Weiqing Xiao",
    "affiliations": [
      "4"
    ],
    "markers": []
  },
  {
    "name": "Weiyi Lin",
    "affiliations": [
      "1"
    ],
    "markers": []
  },
  {
    "name": "Zenan Meng",
    "affiliations": [
      "1"
    ],
    "markers": [
      "‡"
    ]
  },
  {
    "name": "HU WEI",
    "affiliations": [
      "1"
    ],
    "markers": [
      "‡"
    ]
  },
  {
    "name": "Lin Qu",
    "affiliations": [
      "1"
    ],
    "markers": []
  },
  {
    "name": "Zongzheng Zhang",
    "affiliations": [
      "2"
    ],
    "markers": []
  },
  {
    "name": "Hao Zhao",
    "affiliations": [
      "2"
    ],
    "markers": []
  },
  {
    "name": "Chaojian Li",
    "affiliations": [
      "5"
    ],
    "markers": [
      "‡"
    ]
  }
],
  affiliations: [
  {
    "id": "1",
    "name": "Alibaba Group"
  },
  {
    "id": "2",
    "name": "Tsinghua University"
  },
  {
    "id": "3",
    "name": "Nanyang Technological University"
  },
  {
    "id": "4",
    "name": "Nanjing University"
  },
  {
    "id": "5",
    "name": "The Hong Kong University of Science and Technology"
  }
],
  authorNotes: [
  {
    "marker": "*",
    "label": "Equal contribution"
  },
  {
    "marker": "†",
    "label": "Project leader"
  },
  {
    "marker": "‡",
    "label": "Corresponding authors"
  }
],
  links: {
    manuscript: "assets/RFW-S_manuscript.pdf",
    arxiv: "https://arxiv.org/abs/2611.23056",
    code: "https://github.com/RoboFlywheel-com/RoboFlywheel-Soft",
    dataset: "",
    license: "https://github.com/RoboFlywheel-com/RoboFlywheel-Soft/blob/main/LICENSE"
  },
  canonicalUrl: "https://roboflywheel-com.github.io/RoboFlywheel-Soft/",
  citation: `@article{guo2026rfws,
  title={RFW-S: Everyday Deformable Objects with Independent Simulation Verification},
  author={Guo, Jiarui and Zhang, Saining and Wan, Jiaying and Weng, Jiawei and Chen, Junhao and Zhang, Rui and Lyu, Yuanbo and Lee, Zhiheng and Li, Peishuo and Du, Jingrun and Xiao, Weiqing and Lin, Weiyi and Meng, Zenan and Wei, Hu and Qu, Lin and Zhang, Zongzheng and Zhao, Hao and Li, Chaojian},
  journal={arXiv preprint arXiv:2611.23056},
  year={2026}
}`,
  census: [
    {family: "Soft solids", categories: 168, pool: 34641, physical: 21112, runtime: 34502, both: 21103},
    {family: "Cloth sheets", categories: 89, pool: 18347, physical: 18308, runtime: 18097, both: 18064},
    {family: "Shells", categories: 63, pool: 12990, physical: 11415, runtime: 12990, both: 11415},
    {family: "Garments", categories: 94, pool: 19386, physical: 19055, runtime: 19386, both: 19055},
    {family: "Bags", categories: 71, pool: 14636, physical: 13728, runtime: 14611, both: 13728}
  ],
  benchmark: [
    {method: "Fixed command", task: 245, total: 340, percent: 72.1, intact: 0, damage: 0.654},
    {method: "Slip feedback", task: 245, total: 340, percent: 72.1, intact: 0, damage: 0.784},
    {method: "Force prior (local)", task: 200, total: 340, percent: 58.8, intact: 0, damage: 0.478},
    {method: "Privileged reference", task: 200, total: 340, percent: 58.8, intact: 0, damage: 0.493, privileged: true}
  ]
};
